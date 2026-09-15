"""
interface.py - run every agent once, collect logs, no per-agent calls needed.
Assumes all agent files sit next to this one (ppo_simple.py, ppo_icm.py,
td3FINAL.py, td3_rndFINAL.py, sac_main.py, train.py).
"""
import importlib, traceback, json, glob, shutil, os

RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)

# name -> (module, callable). Adjust callable name for td3_rnd once confirmed.
AGENTS = {
    "ppo":         ("ppo_simple",   "train_ppo"),
    "ppo_icm":     ("ppo_icm",      "train_ppo_icm"),
    "td3":         ("td3FINAL",     "train_td3"),
    "td3_rnd":     ("td3_rndFINAL", "train_td3_rnd"),   # confirm real name
    "sac":         ("sac_main",     "main"),
    "model_based": ("train",        "main"),
}

results = {}
for name, (mod_name, fn_name) in AGENTS.items():
    print(f"=== {name} ===")
    try:
        mod = importlib.import_module(mod_name)
        getattr(mod, fn_name)()
        results[name] = "ok"
    except Exception as e:
        traceback.print_exc()
        results[name] = f"failed: {e}"

    # sweep up whatever csv/png each script dropped in cwd, tag with agent name
    for f in glob.glob("*.csv") + glob.glob("*.png"):
        shutil.move(f, os.path.join(RUN_DIR, f"{name}_{f}"))

with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))