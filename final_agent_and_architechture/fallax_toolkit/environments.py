"""Dashboard environment registry; importing this never imports an ML runtime."""
from pathlib import Path
import sys
from .nasim.config import ENVIRONMENT, AGENT, PROFILES

WALKER = "BipedalWalker-v3"
ENVIRONMENTS = {
    WALKER: {"label": "BipedalWalker", "agents": ["ppo", "ppo_icm", "td3", "td3_rnd", "sac", "model_based"],
             "profiles": {"optimized_v1": {"label": "Optimized v1"}, "original": {"label": "Original reward"}},
             "default_profile": "optimized_v1", "minimum_steps": 100000},
    ENVIRONMENT: {"label": "NASim · Pentesting simulation", "agents": [AGENT], "profiles": PROFILES,
                  "default_profile": "nasim_current", "minimum_steps": 1024},
}


def nasim_python(engine):
    # Repository-specific optional runtime; installed wheels use their own Python.
    folder = Path(engine).parent / ".venv-nasim"
    path = folder / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return str(path) if path.is_file() else sys.executable
