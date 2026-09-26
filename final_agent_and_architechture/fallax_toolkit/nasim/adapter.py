"""A simulator-only adapter; never executes commands against real networks."""
import hashlib
import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import nasim
import yaml

from . import notebook_model as model
from .config import PROFILES, SPLITS, reward_metadata


@lru_cache(maxsize=256)
def _scenario_template(path, expected_sha256):
    """Load immutable bank inputs once per worker, before optimization starts."""
    # Bank fingerprints use canonical LF text on both Windows and Linux.
    content = Path(path).read_text(encoding="utf-8").encode("utf-8")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"Scenario contents do not match the bank fingerprint: {path}")
    env = nasim.load(path, fully_obs=False)
    try:
        return deepcopy(env.scenario)
    finally:
        env.close()


def make_environment(spec):
    path = spec.get("path")
    if not isinstance(path, str):
        raise ValueError(f"Invalid scenario path for {spec.get('name')}: {path!r}")
    # Never share a mutable scenario or an environment's episode state.
    template = _scenario_template(path, spec["sha256"])
    return nasim.NASimEnv(deepcopy(template), fully_obs=False)


def create_bank(directory, profile):
    """Retain the notebook's networks and split seeds, changing scan cost only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scan_cost = PROFILES[profile]["scan_cost"]
    banks = {}
    for split, (counts, offset) in SPLITS.items():
        specs = []
        for difficulty, count in enumerate(counts):
            for index in range(count):
                seed = offset + difficulty * 1000 + index
                scenario, honeypot = model.build_random_scenario(seed, difficulty, split)
                for action in model.SCAN_ACTION_NAMES:
                    scenario[action + "_cost"] = scan_cost
                name = f"{split}_d{difficulty}_{index:03d}"
                path = directory / (name + ".yaml")
                content = yaml.safe_dump(scenario, sort_keys=False)
                # Publish complete bank inputs before validation.
                temporary = path.with_suffix(".yaml.tmp")
                temporary.write_text(content, encoding="utf-8")
                os.replace(temporary, path)
                specs.append({"path": str(path.resolve()), "name": name, "seed": seed,
                              "difficulty": difficulty, "split": split, "honeypot": honeypot,
                              "sha256": hashlib.sha256(content.encode()).hexdigest()})
        banks[split] = specs
    return banks


def validate_bank(banks):
    """Parse every generated scenario before a training worker can use it."""
    for specs in banks.values():
        for spec in specs:
            path = spec.get("path")
            if not isinstance(path, str) or not Path(path).is_file():
                raise ValueError(f"Scenario path is invalid for {spec.get('name')}: {path!r}")
            try:
                env = make_environment(spec)
                env.close()
            except Exception as exc:
                raise ValueError(f"Scenario bank validation failed for {spec.get('name')}: {exc}") from exc


def bank_fingerprint(banks):
    content = [(split, s["name"], s["sha256"]) for split, specs in banks.items() for s in specs]
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()


def create_models(seed=2026):
    torch.manual_seed(seed)
    encoder = model.NetworkStateEncoder(in_channels=model.NODE_FEATURE_DIM,
                    hidden_channels=16, out_channels=16, edge_dim=model.EDGE_FEATURE_DIM, num_heads=4)
    actor = model.ActorNetwork(model.action_types_per_host, embedding_size=16)
    critic = model.Critic(32)
    return encoder, actor, critic


class Episode:
    """Exactly one active episode per process (notebook trackers are globals).

    Frozen PyG inputs and action masks are returned by choose(). Numerical
    observations alone cannot reconstruct decisions with history-dependent memory.
    """
    def __init__(self, spec, profile, seed):
        self.spec, self.profile = spec, profile
        path = spec.get("path")
        if not isinstance(path, str):
            raise ValueError(f"Scenario path for {spec.get('name')} is not a string: {path!r}")
        try:
            self.env = make_environment(spec)
        except Exception as exc:
            raise ValueError(f"Could not load NASim scenario {spec.get('name')} at {path}: {exc}") from exc
        self.obs, _ = self.env.reset(seed=seed)
        # NASim 0.12 also uses NumPy's legacy RNG in transition sampling.
        np.random.seed(seed)
        model.reset_episode_memory()
        self.reference = model.get_reference_value(self.env, spec)
        self.steps = 0
        self.last_action_key = None
        self.repeat_count = 0

    def graph(self):
        graph = model.build_graph(self.obs, self.env)
        pyg = model.convert_to_PyG(graph)
        mask = model.build_action_mask(graph["discovered_hosts"], model.action_types,
                   model.node_service_state, model.node_process_state, model.node_scan_state,
                   model.known_host_states)
        return graph, pyg, mask

    def choose(self, encoder, actor, critic, deterministic=False):
        graph, pyg, mask = self.graph()
        nodes = encoder.gnn(pyg.x, pyg.edge_index, pyg.edge_attr)
        embedding = torch.cat([nodes.mean(dim=0), nodes.max(dim=0).values])
        chosen, log_prob, distribution = actor(nodes, mask, deterministic=deterministic)
        index = chosen.item()
        target = graph["discovered_hosts"][index // model.action_types_per_host]
        action_name = model.action_types[index % model.action_types_per_host]
        action = model.build_nasim_action(action_name, target, self.env)
        valid_count = int(mask.sum())
        return {"action": action, "action_name": action_name, "target": list(target),
                "chosen": chosen, "log_prob": log_prob, "value": critic(embedding).squeeze(),
                "embedding": embedding.detach().clone(), "pyg": pyg.clone(), "mask": mask.clone(),
                "valid_actions": valid_count,
                "entropy": float(distribution.entropy().detach()),
                "normalized_entropy": float(distribution.entropy().detach()) / np.log(valid_count) if valid_count > 1 else None}

    def _progress(self):
        addresses = self.env.current_state.host_num_map
        normal = [a for a in addresses if a != tuple(self.spec["honeypot"])]
        return {
            "compromised": sum(self.env.current_state.host_compromised(a) for a in normal),
            "root": sum(self.env.current_state.get_host(a).access >= model.AccessLevel.ROOT for a in normal),
            "discovered": sum(bool(self.env.current_state.get_host(a).discovered) for a in addresses),
            "knowledge": sum(v.get("known", 0) != 0 for v in model.node_service_state.values())
                         + sum(v.get("known", 0) != 0 for v in model.node_process_state.values())
                         + sum(bool(v) for v in model.node_scan_state.values()),
        }

    def step(self, decision):
        action = decision["action"]
        before = self._progress()
        host = self.env.current_state.get_host(action.target)
        was_compromised = bool(host.compromised)
        had_root = host.access >= model.AccessLevel.ROOT
        known_service = known_process = None
        if action.is_exploit():
            known_service = model.node_service_state.get((action.target, model.service_to_idx[action.service]), {}).get("known", 0)
        elif isinstance(action, model.PrivilegeEscalation):
            known_process = model.node_process_state.get((action.target, model.PROCESS_TO_IDX[action.process]), {}).get("known", 0)
        obs, native, terminated, truncated, info = self.env.step(action)
        outcome = model.update_trackers(action, info, self.env, model.edge_tracker,
                          model.node_service_state, model.node_process_state, model.node_scan_state)
        if PROFILES[self.profile]["shaping"]:
            reward, components = model.compute_reward(self.env, action, info, outcome, self.spec,
                        was_compromised, had_root, self.reference, reward_from_env=native,
                        known_before_service=known_service, known_before_process=known_process)
        else:
            reward = native
            components = {"native_value": float(info.get("value", 0.0)), "action_cost": -float(action.cost), "total": float(reward)}
        self.obs = obs
        self.steps += 1
        boundary_reason = "goal" if terminated else "time_limit" if truncated else None
        if not terminated and not truncated:
            try:
                self.graph()
            except RuntimeError as exc:
                if "no valid actions" not in str(exc):
                    raise
                # Preserve the notebook mask, but end exhausted episodes explicitly
                # instead of crashing or inventing a valid action.
                truncated = True
                boundary_reason = "no_valid_actions"
        after = self._progress()
        changes = {key: int(after[key] - before[key]) for key in before}
        made_progress = any(v > 0 for v in changes.values())
        key = (decision["action_name"], tuple(decision["target"]))
        self.repeat_count = self.repeat_count + 1 if key == self.last_action_key else 0
        self.last_action_key = key
        honeypot = tuple(action.target) == tuple(self.spec["honeypot"]) and (action.is_exploit() or isinstance(action, model.PrivilegeEscalation))
        row = {"episode_step": self.steps, "scenario": self.spec["name"], "difficulty": self.spec["difficulty"],
               "action": decision["action_name"], "target": decision["target"],
               "outcome": outcome.value if outcome else "unknown", "success": bool(info.get("success", False)),
               "reward": float(reward), "native_reward": float(native), "reward_components": components,
               "value": float(decision["value"]), "normalized_entropy": decision["normalized_entropy"],
               "valid_actions": decision["valid_actions"], "repeat_count": self.repeat_count,
               "progress": changes, "positive_reward_without_progress": bool(reward > 0 and not made_progress),
               "honeypot_targeted": bool(honeypot), "goal_reached": bool(self.env.goal_reached()),
               "terminated": bool(terminated), "truncated": bool(truncated), "boundary_reason": boundary_reason}
        return float(reward), bool(terminated), bool(truncated), row

    def close(self):
        self.env.close()
