"""
dry_run.py - sanity-check that every agent starts and runs without error.
Each agent is launched in its own process and killed after TIMEOUT seconds
(most agents loop for millions of steps with no built-in short-run flag,
so a timeout is the simplest universal way to smoke-test all of them).
"""
import multiprocessing as mp
import importlib
import traceback

TIMEOUT = 20  # seconds per agent

AGENTS = {
    "ppo":         ("ppo_simple",   "train_ppo"),
    "ppo_icm":     ("ppo_icm",      "train_ppo_icm"),
    "td3":         ("td3FINAL",     "train_td3"),
    "td3_rnd":     ("td3_rndFINAL", "train"),
    "sac":         ("sac_main",     "main"),
    "model_based": ("train",        "main"),
}


def _run(mod_name, fn_name):
    try:
        mod = importlib.import_module(mod_name)
        getattr(mod, fn_name)()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    for name, (mod_name, fn_name) in AGENTS.items():
        print(f"\n=== {name} (max {TIMEOUT}s) ===", flush=True)
        p = mp.Process(target=_run, args=(mod_name, fn_name))
        p.start()
        p.join(TIMEOUT)
        if p.is_alive():
            p.terminate()
            p.join()
            print(f"[{name}] ran cleanly for {TIMEOUT}s, killed (expected).")
        else:
            status = "exited on its own (check output above for errors)" if p.exitcode == 0 else f"CRASHED (exit {p.exitcode})"
            print(f"[{name}] {status}")