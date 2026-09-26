"""Loopback-only dashboard, static assets, read-only reports and owned jobs."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import webbrowser

AGENTS = {
    "ppo": {"label": "PPO", "family": "On-policy", "description": "Clipped policy optimization", "steps": 2500000},
    "ppo_icm": {"label": "PPO + ICM", "family": "Intrinsic curiosity", "description": "Prediction-driven exploration", "steps": 1000000},
    "td3": {"label": "TD3", "family": "Off-policy", "description": "Twin delayed deterministic policy", "steps": 1000000},
    "td3_rnd": {"label": "TD3 + RND", "family": "Intrinsic curiosity", "description": "Random network distillation", "steps": 1000000},
    "sac": {"label": "SAC", "family": "Maximum entropy", "description": "Soft actor-critic", "steps": 500000},
    "model_based": {"label": "Model-based PPO", "family": "Model-based", "description": "Learned dynamics and imagined rollouts", "steps": 3072000},
}
STATIC = Path(__file__).parent / "static"
ENGINE = Path(__file__).resolve().parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(clean_json(value), indent=2), encoding="utf-8")
    temp.replace(path)


class Controller:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.jobs = {}
        self.processes = {}

    def experiment(self, key):
        if key == "legacy":
            return self.root
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", key):
            raise ValueError("Use 1–64 letters, numbers, underscores or hyphens for the experiment name.")
        path = (self.root / "experiments" / key).resolve()
        if path.parent != (self.root / "experiments").resolve():
            raise ValueError("Experiment must remain in the experiments directory.")
        return path

    def experiments(self):
        paths = []
        if (self.root / "runs/hackability_report.json").exists():
            paths.append(("legacy", self.root))
        directory = self.root / "experiments"
        if directory.exists():
            for p in sorted(directory.iterdir()):
                if p.is_dir() and (p / "experiment.json").exists():
                    try:
                        self.experiment(p.name)
                    except ValueError:
                        continue
                    paths.append((p.name, p))
        items = []
        for key, path in paths:
            manifest = read_json(path / "experiment.json") if key != "legacy" else {}
            summary = read_json(path / "summary.json")
            report_path = path / "runs/hackability_report.json"
            data = read_json(report_path)
            items.append({"id": key, "name": "Original baseline" if key == "legacy" else key,
                          "profile": manifest.get("reward", {}).get("profile", "original"),
                          "manifest": manifest, "summary": summary,
                          "report_available": bool(data), "agents": [n for n in AGENTS if n in data],
                          "updated": datetime.fromtimestamp(report_path.stat().st_mtime, timezone.utc).isoformat() if report_path.exists() else None,
                          "readonly": key == "legacy"})
        return items

    def state(self):
        with self.lock:
            jobs = [dict(j) for j in self.jobs.values()]
        # Prior sessions are history only; never adopt/kill an unknown process.
        history_dir = self.root / ".fallax/jobs"
        if history_dir.exists():
            known = {j["id"] for j in jobs}
            for p in sorted(history_dir.glob("*.json"), reverse=True)[:30]:
                job = read_json(p)
                if job and job.get("id") not in known:
                    if job.get("status") in ("running", "queued", "stopping"):
                        job["status"] = "detached"
                    jobs.append(job)
        dependencies = {name: importlib.util.find_spec(module) is not None for name, module in
                        (("NumPy", "numpy"), ("PyTorch", "torch"), ("Gymnasium", "gymnasium"), ("Box2D", "Box2D"), ("Matplotlib", "matplotlib"))}
        return {"experiments": self.experiments(), "agents": AGENTS, "jobs": jobs,
                "root": str(self.root), "python": sys.executable, "dependencies": dependencies,
                "token": self.token, "version": "0.1.0"}

    def save_job(self, job):
        write_json(self.root / ".fallax/jobs" / (job["id"] + ".json"), job)

    def start(self, data):
        mode = data.get("mode")
        if mode not in ("train", "report"):
            raise ValueError("Choose training or report generation.")
        agents = data.get("agents")
        if not isinstance(agents, list) or not agents or any(not isinstance(n, str) or n not in AGENTS for n in agents) or len(set(agents)) != len(agents):
            raise ValueError("Select at least one unique supported agent.")
        key = data.get("experiment")
        if key == "legacy":
            raise ValueError("The original baseline is read-only. Create a new experiment.")
        output = self.experiment(key)
        steps = data.get("steps")
        if steps is not None and (type(steps) is not int or not 100000 <= steps <= 10000000):
            raise ValueError("Training budget must be 100,000–10,000,000 steps, or default.")
        profile = data.get("profile", "optimized_v1")
        if profile not in ("original", "optimized_v1"):
            raise ValueError("Unknown reward profile.")
        with self.lock:
            if any(j["status"] in ("queued", "running", "stopping") for j in self.jobs.values()):
                raise ValueError("A dashboard job is already active. Wait for it or stop it first.")
            if mode == "train":
                if output.exists() and any(output.iterdir()):
                    raise ValueError("This experiment already contains files. Use a new name to keep existing training safe.")
                # Imports happen only at explicit launch, so report viewing needs no ML runtime.
                from reward_wrapper import reward_metadata
                manifest = {"environment": "BipedalWalker-v3", "reward": reward_metadata(profile),
                            "agents": agents, "initialization": "fresh", "created": now(),
                            "python": sys.executable, "requested_steps": steps,
                            "budgets": {a: steps or AGENTS[a]["steps"] for a in agents}}
                output.mkdir(parents=True, exist_ok=True)
                write_json(output / "experiment.json", manifest)
            else:
                if not (output / "experiment.json").exists():
                    raise ValueError("Experiment manifest is missing.")
                summary = read_json(output / "summary.json")
                if any(summary.get(a) != "ok" for a in agents):
                    raise ValueError("Only agents marked complete in the training summary can be evaluated.")
                profile = read_json(output / "experiment.json")["reward"]["profile"]
            job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
            job = {"id": job_id, "mode": mode, "experiment": key, "agents": agents,
                   "steps": steps, "profile": profile, "status": "queued", "created": now(),
                   "current_agent": None, "completed_agents": [], "error": None}
            self.jobs[job_id] = job
            self.save_job(job)
            threading.Thread(target=self._run, args=(job_id, output), daemon=True).start()
            return dict(job)

    def _run(self, job_id, output):
        job = self.jobs[job_id]
        log_path = self.root / ".fallax/jobs" / (job_id + ".log")
        environment = os.environ.copy()
        environment.update(WALKER_REWARD_PROFILE=job["profile"], MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
                           PYTHONPATH=str(ENGINE) + os.pathsep + environment.get("PYTHONPATH", ""))
        summary = read_json(output / "summary.json")
        try:
            with log_path.open("a", encoding="utf-8") as log:
                groups = [[a] for a in job["agents"]] if job["mode"] == "train" else [job["agents"]]
                for agents in groups:
                    with self.lock:
                        if job["status"] == "stopping":
                            break
                        job.update(status="running", current_agent=", ".join(agents))
                        self.save_job(job)
                        command = [sys.executable, "-u", "-B", "-m", "fallax_toolkit.worker", job["mode"], "--agents", *agents]
                        if job["mode"] == "train":
                            command += ["--steps", str(job["steps"] or AGENTS[agents[0]]["steps"])]
                        process = subprocess.Popen(command, cwd=output, env=environment,
                                                   stdout=log, stderr=subprocess.STDOUT,
                                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        self.processes[job_id] = process
                    returncode = process.wait()
                    with self.lock:
                        self.processes.pop(job_id, None)
                        stopped = job["status"] == "stopping"
                        if job["mode"] == "train":
                            summary[agents[0]] = "stopped" if stopped else "ok" if returncode == 0 else f"failed: exit {returncode}"
                            write_json(output / "summary.json", summary)
                        if stopped:
                            break
                        if returncode != 0:
                            raise RuntimeError(f"{job['current_agent']} exited with code {returncode}. See the job log.")
                        job["completed_agents"].extend(agents)
                        self.save_job(job)
                with self.lock:
                    job["status"] = "stopped" if job["status"] == "stopping" else "completed"
        except Exception as exc:
            with self.lock:
                job.update(status="failed", error=str(exc))
        finally:
            with self.lock:
                job["finished"] = now()
                self.save_job(job)

    def stop(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job["status"] not in ("running", "queued"):
                raise ValueError("Only an active job started by this dashboard can be stopped.")
            job["status"] = "stopping"
            process = self.processes.get(job_id)
            if process and process.poll() is None:
                process.terminate()
            self.save_job(job)
            return dict(job)

    def log(self, job_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
            raise ValueError("Invalid job ID.")
        path = self.root / ".fallax/jobs" / (job_id + ".log")
        if not path.exists():
            return "Waiting for job output."
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 48000))
            return handle.read().decode("utf-8", errors="replace")


def make_handler(controller):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, value, content_type="application/json"):
            body = json.dumps(clean_json(value)).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def do_GET(self):
            if not self.valid_host():
                return self.respond(403, {"error": "Local requests only."})
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/state":
                    return self.respond(200, controller.state())
                if parsed.path == "/api/report":
                    key = query.get("experiment", ["legacy"])[0]
                    path = controller.experiment(key) / "runs/hackability_report.json"
                    if not path.exists():
                        return self.respond(404, {"error": "No report yet. Evaluate completed agents first."})
                    return self.respond(200, read_json(path))
                if parsed.path == "/api/log":
                    return self.respond(200, {"text": controller.log(query.get("job", [""])[0])})
                names = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
                if parsed.path not in names:
                    return self.respond(404, {"error": "Not found."})
                path = STATIC / names[parsed.path]
                mime = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}[path.suffix]
                return self.respond(200, path.read_bytes(), mime)
            except (ValueError, OSError) as exc:
                return self.respond(400, {"error": str(exc)})

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if not self.valid_host() or (origin and origin not in allowed) or self.headers.get("X-Fallax-Token") != controller.token:
                return self.respond(403, {"error": "Invalid local session. Reload the dashboard."})
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 16384:
                    raise ValueError("Invalid request size.")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Expected an object.")
                if self.path == "/api/jobs":
                    return self.respond(202, controller.start(data))
                if self.path == "/api/stop":
                    return self.respond(200, controller.stop(data.get("id", "")))
                return self.respond(404, {"error": "Not found."})
            except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
                return self.respond(400, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Fallax local research dashboard")
    parser.add_argument("--root", type=Path, help="Toolkit data directory containing runs/ and experiments/")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    root = args.root or (ENGINE if (ENGINE / "interface.py").exists() and (ENGINE / "runs").exists() else Path.cwd() / "fallax-data")
    controller = Controller(root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(controller))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Fallax dashboard: {url}\nData directory: {controller.root}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with controller.lock:
            for key, job in controller.jobs.items():
                if job["status"] in ("queued", "running"):
                    controller.stop(key)
        server.server_close()


if __name__ == "__main__":
    main()
