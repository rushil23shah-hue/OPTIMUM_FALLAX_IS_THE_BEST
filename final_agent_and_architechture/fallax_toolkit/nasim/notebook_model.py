"""Definitions copied from the fellow mentee's notebook for isolated NASim use.

Original notebook is not executed or modified. Source SHA256:
4ba93663cde54ab87a7719517b56bbb19877358f9c256e884fa102f989a966e6
Only definitions/constants are included; installation, generation and training
cells do not execute on import. One active NASim memory context per worker.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from enum import Enum
from nasim.envs.action import SubnetScan, ServiceScan, OSScan, Exploit, PrivilegeEscalation, ProcessScan
from nasim.envs.host_vector import HostVector
from nasim.envs.utils import AccessLevel


# Copied definitions from notebook cell 8 (including markdown cells).
OS_VOCAB = ["linux", "windows"]

SERVICE_VOCAB = ["ssh", "ftp", "http"]

PROCESS_VOCAB = ["tomcat", "daclsvc"]

MAX_SUBNET_ID = 4

MAX_HOST_ID = 4

TOPOLOGY_TEMPLATES = [
    # Internet -> DMZ; DMZ branches to sensitive/user; user -> final sensitive.
    {
        "edges": [(0, 1), (1, 2), (1, 3), (2, 3), (3, 4)],
        "parents": {1: 0, 2: 1, 3: 1, 4: 3},
    },
    # Deeper chain.
    {
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "parents": {1: 0, 2: 1, 3: 2, 4: 3},
    },
    # User subnet is the central pivot.
    {
        "edges": [(0, 1), (1, 3), (3, 2), (3, 4)],
        "parents": {1: 0, 3: 1, 2: 3, 4: 3},
    },
]

def _topology_matrix(edges):
    matrix = np.eye(5, dtype=int)
    for left, right in edges:
        matrix[left, right] = 1
        matrix[right, left] = 1
    return matrix.tolist()

def _compatible_os(service, rng):
    if service == "ssh":
        return "linux"
    if service == "ftp":
        return "windows"
    return str(rng.choice(OS_VOCAB))

def build_random_scenario(seed, difficulty, split):
    rng = np.random.default_rng(seed)
    user_hosts = int(rng.integers(2, 4 + difficulty))
    user_hosts = min(user_hosts, 5)
    subnets = [1, 1, user_hosts, 1]

    if difficulty == 0:
        template_index = 0
    else:
        template_index = int(rng.integers(0, len(TOPOLOGY_TEMPLATES)))
    template = TOPOLOGY_TEMPLATES[template_index]
    topology = _topology_matrix(template["edges"])

    gateway_service = {
        subnet: str(rng.choice(SERVICE_VOCAB))
        for subnet in range(1, 5)
    }

    if difficulty == 0:
        exploit_probs = {service: 0.9 for service in SERVICE_VOCAB}
        privesc_prob = 1.0
        exploit_costs = {service: 1.0 for service in SERVICE_VOCAB}
    elif difficulty == 1:
        exploit_probs = {
            service: float(rng.choice([0.65, 0.8, 0.9]))
            for service in SERVICE_VOCAB
        }
        privesc_prob = float(rng.choice([0.8, 0.9, 1.0]))
        exploit_costs = {
            service: float(rng.choice([1.0, 2.0]))
            for service in SERVICE_VOCAB
        }
    else:
        exploit_probs = {
            service: float(rng.choice([0.4, 0.55, 0.7, 0.9]))
            for service in SERVICE_VOCAB
        }
        privesc_prob = float(rng.choice([0.7, 0.85, 1.0]))
        exploit_costs = {
            service: float(rng.choice([1.0, 2.0, 3.0]))
            for service in SERVICE_VOCAB
        }

    firewall = {}
    for left, right in template["edges"]:
        for src, dst in ((left, right), (right, left)):
            if difficulty == 0:
                allowed = list(SERVICE_VOCAB)
            else:
                count = 2 if difficulty == 1 else int(rng.integers(1, 3))
                allowed = [
                    str(service)
                    for service in rng.choice(
                        SERVICE_VOCAB, size=count, replace=False
                    )
                ]
            if template["parents"].get(dst) == src:
                allowed = list(dict.fromkeys(allowed + [gateway_service[dst]]))
            firewall[str((src, dst))] = allowed

    addresses = [(1, 0), (2, 0)]
    addresses.extend((3, host_id) for host_id in range(user_hosts))
    addresses.append((4, 0))
    sensitive_hosts = {(2, 0): 20, (4, 0): 20}
    honeypot_candidates = [(3, host_id) for host_id in range(user_hosts)]
    honeypot = honeypot_candidates[int(rng.integers(0, len(honeypot_candidates)))]

    host_configurations = {}
    for address in addresses:
        subnet = address[0]
        required_service = gateway_service[subnet]
        host_os = _compatible_os(required_service, rng)

        services = [required_service]
        extra_count = int(rng.integers(0, 1 + difficulty))
        extra_count = min(extra_count, len(SERVICE_VOCAB) - 1)
        extras = [
            str(service)
            for service in rng.choice(
                SERVICE_VOCAB, size=extra_count, replace=False
            )
        ]
        services = list(dict.fromkeys(services + extras))

        required_process = "tomcat" if host_os == "linux" else "daclsvc"
        processes = [required_process]
        if difficulty >= 1 and rng.random() < 0.25:
            processes = list(PROCESS_VOCAB)

        if address in sensitive_hosts:
            value = sensitive_hosts[address]
        elif address == honeypot:
            value = -8
        else:
            value = int(rng.integers(2, 8))

        host_configurations[str(address)] = {
            "os": host_os,
            "services": services,
            "processes": processes,
            "value": value,
        }

    scenario = {
        "subnets": subnets,
        "topology": topology,
        "sensitive_hosts": {
            str(address): value for address, value in sensitive_hosts.items()
        },
        "os": list(OS_VOCAB),
        "services": list(SERVICE_VOCAB),
        "processes": list(PROCESS_VOCAB),
        "exploits": {
            "e_ssh": {
                "service": "ssh", "os": "linux",
                "prob": exploit_probs["ssh"],
                "cost": exploit_costs["ssh"], "access": "user",
            },
            "e_ftp": {
                "service": "ftp", "os": "windows",
                "prob": exploit_probs["ftp"],
                "cost": exploit_costs["ftp"], "access": "user",
            },
            "e_http": {
                "service": "http", "os": "None",
                "prob": exploit_probs["http"],
                "cost": exploit_costs["http"], "access": "user",
            },
        },
        "privilege_escalation": {
            "pe_tomcat": {
                "process": "tomcat", "os": "linux",
                "prob": privesc_prob, "cost": 1.0, "access": "root",
            },
            "pe_daclsvc": {
                "process": "daclsvc", "os": "windows",
                "prob": privesc_prob, "cost": 1.0, "access": "root",
            },
        },
        "service_scan_cost": 0.0,
        "os_scan_cost": 0.0,
        "subnet_scan_cost": 0.0,
        "process_scan_cost": 0.0,
        "host_configurations": host_configurations,
        "firewall": firewall,
        "step_limit": 400,
    }
    return scenario, honeypot

# Copied definitions from notebook cell 10 (including markdown cells).
class ActionOutcome(Enum):
    SUCCESS = "success"
    MISMATCH = "mismatch"
    BLOCKED = "blocked"
    PROB_FAIL = "prob_fail"
    PERMISSION = "permission"
    HONEYPOT = "honeypot"

def classify_outcome(action, info, env):
    if info.get("success", False):
        return ActionOutcome.SUCCESS
    target = action.target
    service = action.service
    if not env.current_state.host_is_running_service(target, service):
        return ActionOutcome.MISMATCH
    if not env.network.traffic_permitted(env.current_state, target, service):
        return ActionOutcome.BLOCKED
    if info.get("permission_error", False):
        return ActionOutcome.MISMATCH   
    return ActionOutcome.PROB_FAIL

def classify_outcome_privesc(action, info, env):
    if info.get("success", False):
        return ActionOutcome.SUCCESS
    target = action.target
    process = action.process
    host = env.current_state.get_host(target)
    if not host.is_running_process(process):
        return ActionOutcome.MISMATCH
    if action.os is not None and not host.is_running_os(action.os):
        return ActionOutcome.MISMATCH
    if host.access < action.req_access:
        return ActionOutcome.PERMISSION
    return ActionOutcome.PROB_FAIL

# Copied definitions from notebook cell 15 (including markdown cells).
def get_reference_value(env, spec):
    """A per-scenario baseline so bonus floors scale with this network, not a hardcoded number."""
    values = [
        env.current_state.get_host(addr).value
        for addr in env.current_state.host_num_map
        if addr != spec["honeypot"]
    ]
    return max(sum(values) / len(values), 1.0) if values else 1.0

SCAN_INFORMED_EXPLOIT_BONUS = 1.25

BLIND_EXPLOIT_PENALTY = -1.75

SCAN_INFORMED_PRIVESC_BONUS = 2.5

BLIND_PRIVESC_PENALTY = -3.5

def compute_reward(env, action, info, outcome, current_spec,
                    was_compromised, had_root_access, reference_value,
                    reward_from_env=0.0,
                    known_before_service=None, known_before_process=None):
    reward = reward_from_env - info.get("value", 0.0)  # strip native value credit — avoid double count
    components = {"base_cost": reward}

    outcome_penalties = {
        ActionOutcome.SUCCESS: 0.0,
        ActionOutcome.MISMATCH: -0.2,
        ActionOutcome.BLOCKED: -0.1,
        ActionOutcome.PROB_FAIL: -0.05,
        ActionOutcome.PERMISSION: -0.15,
    }
    outcome_pen = outcome_penalties.get(outcome, 0.0)
    if outcome_pen != 0.0:
        reward += outcome_pen
        components["outcome_penalty"] = outcome_pen

    host_value = env.current_state.get_host(action.target).value if hasattr(action, "target") else 0
    is_priv = isinstance(action, PrivilegeEscalation)

    if action.is_exploit() and info.get("success") and not was_compromised:
        bonus = max(host_value * 1.5, 0.5 * reference_value)
        reward += bonus
        components["exploit_success_bonus"] = bonus
    if is_priv and info.get("success") and not had_root_access:
        bonus = max(host_value * 1.5, 0.7 * reference_value)
        reward += bonus
        components["privesc_success_bonus"] = bonus

    if action.is_exploit() and known_before_service is not None:
        if known_before_service == 1:
            reward += SCAN_INFORMED_EXPLOIT_BONUS
            components["scan_informed_exploit_bonus"] = SCAN_INFORMED_EXPLOIT_BONUS
        elif known_before_service == 0:
            reward += BLIND_EXPLOIT_PENALTY
            components["blind_exploit_penalty"] = BLIND_EXPLOIT_PENALTY

    if is_priv and known_before_process is not None:
        if known_before_process == 1:
            reward += SCAN_INFORMED_PRIVESC_BONUS
            components["scan_informed_privesc_bonus"] = SCAN_INFORMED_PRIVESC_BONUS
        elif known_before_process == 0:
            reward += BLIND_PRIVESC_PENALTY
            components["blind_privesc_penalty"] = BLIND_PRIVESC_PENALTY

    newly_discovered = info.get("newly_discovered", {})
    actually_new = sum(1 for v in newly_discovered.values() if v)
    if getattr(action, "name", "") == "subnet_scan" and info.get("success") and actually_new:
        bonus = 1.0 * actually_new
        reward += bonus
        components["discovery_bonus"] = bonus

    components["total"] = reward
    return reward, components

# Copied definitions from notebook cell 17 (including markdown cells).
known_host_states = {}

node_service_state = {}

node_process_state = {}

node_scan_state = {}

edge_tracker = {}

host_suspicion = {}

SCAN_ACTION_NAMES = ("service_scan", "os_scan", "subnet_scan", "process_scan")

def reset_episode_memory():
    known_host_states.clear()
    node_service_state.clear()
    node_process_state.clear()
    node_scan_state.clear()
    edge_tracker.clear()
    host_suspicion.clear()

def update_node_tracker(host, service_idx, outcome):
    key = (host, service_idx)
    entry = node_service_state.get(key, {"known": 0, "fail_count": 0, "attempts": 0})
    entry["attempts"] += 1
    if outcome == ActionOutcome.SUCCESS:
        entry["known"] = 1
    elif outcome in (ActionOutcome.MISMATCH, ActionOutcome.BLOCKED):
        entry["known"] = -1
    elif outcome == ActionOutcome.PROB_FAIL:
        entry["fail_count"] += 1
    node_service_state[key] = entry
    return entry

def update_node_process_tracker(host, process_idx, outcome):
    key = (host, process_idx)
    entry = node_process_state.get(key, {"known": 0, "fail_count": 0, "attempts": 0})
    entry["attempts"] += 1
    if outcome == ActionOutcome.SUCCESS:
        entry["known"] = 1
    elif outcome == ActionOutcome.MISMATCH:
        entry["known"] = -1
    elif outcome == ActionOutcome.PROB_FAIL:
        entry["fail_count"] += 1
    node_process_state[key] = entry
    return entry

def _scan_name(action):
    if isinstance(action, ServiceScan):
        return "service_scan"
    if isinstance(action, OSScan):
        return "os_scan"
    if isinstance(action, SubnetScan):
        return "subnet_scan"
    if isinstance(action, ProcessScan):
        return "process_scan"
    return None

def _scan_outcome(info):
    if info.get("success", False):
        return ActionOutcome.SUCCESS
    if info.get("permission_error", False):
        return ActionOutcome.PERMISSION
    if info.get("connection_error", False):
        return ActionOutcome.BLOCKED
    return ActionOutcome.PROB_FAIL

def _record_service_knowledge(host, env):
    for service, service_idx in service_to_idx.items():
        entry = node_service_state.get(
            (host, service_idx), {"known": 0, "fail_count": 0, "attempts": 0}
        )
        entry["known"] = 1 if env.current_state.host_is_running_service(host, service) else -1
        node_service_state[(host, service_idx)] = entry

def _record_process_knowledge(host, env):
    host_state = env.current_state.get_host(host)
    for process, process_idx in PROCESS_TO_IDX.items():
        entry = node_process_state.get(
            (host, process_idx), {"known": 0, "fail_count": 0, "attempts": 0}
        )
        entry["known"] = 1 if host_state.is_running_process(process) else -1
        node_process_state[(host, process_idx)] = entry

def get_compromised_subnets(env):
    return {
        address[0]
        for address in env.current_state.host_num_map
        if env.current_state.host_compromised(address)
    }

def update_edge_tracker_for_action(edge_tracker, action, env, outcome):
    if outcome not in (ActionOutcome.BLOCKED, ActionOutcome.SUCCESS, ActionOutcome.PROB_FAIL):
        return
    dst_subnet = action.target[0]
    service_idx = service_to_idx[action.service]
    for src_subnet in get_compromised_subnets(env):
        if src_subnet == dst_subnet:
            continue
        permitted = env.network.subnet_traffic_permitted(
            src_subnet, dst_subnet, action.service
        )
        key = (src_subnet, dst_subnet, service_idx)
        entry = edge_tracker.get(key, {"outcome": 0, "attempts": 0})
        entry["attempts"] += 1
        entry["outcome"] = 1 if permitted else -1
        edge_tracker[key] = entry

def update_trackers(action, info, env, edge_tracker, node_service_state,
                    node_process_state, node_scan_state):
    scan_name = _scan_name(action)
    if scan_name is not None:
        outcome = _scan_outcome(info)
        if outcome == ActionOutcome.SUCCESS:
            node_scan_state[(action.target, scan_name)] = True
            if isinstance(action, ServiceScan):
                _record_service_knowledge(action.target, env)
            elif isinstance(action, ProcessScan):
                _record_process_knowledge(action.target, env)
        return outcome

    if isinstance(action, Exploit):
        outcome = classify_outcome(action, info, env)
        update_node_tracker(action.target, service_to_idx[action.service], outcome)
        update_edge_tracker_for_action(edge_tracker, action, env, outcome)
        if outcome == ActionOutcome.SUCCESS:
            _record_service_knowledge(action.target, env)
            node_scan_state[(action.target, "service_scan")] = True
            node_scan_state[(action.target, "os_scan")] = True
        return outcome

    if isinstance(action, PrivilegeEscalation):
        outcome = classify_outcome_privesc(action, info, env)
        update_node_process_tracker(
            action.target, PROCESS_TO_IDX[action.process], outcome
        )
        if outcome == ActionOutcome.SUCCESS:
            _record_process_knowledge(action.target, env)
            node_scan_state[(action.target, "process_scan")] = True
            node_scan_state[(action.target, "os_scan")] = True
        return outcome

    return None

# Copied definitions from notebook cell 19 (including markdown cells).
def read_host_observation(host_slice, address):
    readable = HostVector.get_readable(host_slice)
    return {
        "address": address,
        "compromised": float(readable["Compromised"]),
        "reachable": float(readable["Reachable"]),
        "discovered": float(readable["Discovered"]),
        "value": float(readable["Value"]),
        "discovery_value": float(readable["Discovery Value"]),
        "access": float(readable["Access"]),
        "os": np.asarray([readable[name] for name in OS_VOCAB], dtype=np.float32),
        "services": np.asarray([readable[name] for name in SERVICE_VOCAB], dtype=np.float32),
        "processes": np.asarray([readable[name] for name in PROCESS_VOCAB], dtype=np.float32),
    }

def merge_host_observation(previous, current):
    merged = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in previous.items()
    }
    for key in ("compromised", "reachable", "discovered", "access"):
        merged[key] = max(previous[key], current[key])
    for key in ("value", "discovery_value"):
        if current[key] != 0:
            merged[key] = current[key]
    for key in ("os", "services", "processes"):
        if np.any(current[key] != 0):
            merged[key] = current[key].copy()
    return merged

def _tracker_features(host, state_map, count):
    features = []
    for item_idx in range(count):
        entry = state_map.get((host, item_idx), {})
        features.extend([
            float(entry.get("known", 0)),
            float(np.log1p(entry.get("fail_count", 0))),
            float(np.log1p(entry.get("attempts", 0))),
        ])
    return features

SEMANTIC_HOST_FEATURE_DIM = (
    2 + 6 + len(OS_VOCAB) + len(SERVICE_VOCAB) + len(PROCESS_VOCAB)
)

NODE_FEATURE_DIM = (
    SEMANTIC_HOST_FEATURE_DIM
    + 3 * len(SERVICE_VOCAB)
    + 3 * len(PROCESS_VOCAB)
    + len(SCAN_ACTION_NAMES)
)

EDGE_FEATURE_DIM = len(SERVICE_VOCAB) + 1

def host_feature_vector(host, memory):
    subnet, host_id = host
    base = np.concatenate([
        np.asarray([
            subnet / MAX_SUBNET_ID,
            host_id / max(MAX_HOST_ID, 1),
            memory["compromised"],
            memory["reachable"],
            memory["discovered"],
            memory["value"] / 20.0,
            memory["discovery_value"] / 20.0,
            memory["access"] / float(AccessLevel.ROOT),
        ], dtype=np.float32),
        memory["os"],
        memory["services"],
        memory["processes"],
    ])
    service_features = _tracker_features(
        host, node_service_state, len(SERVICE_VOCAB)
    )
    process_features = _tracker_features(
        host, node_process_state, len(PROCESS_VOCAB)
    )
    scan_features = [
        float(node_scan_state.get((host, scan_name), False))
        for scan_name in SCAN_ACTION_NAMES
    ]
    return np.concatenate([
        base,
        np.asarray(service_features, dtype=np.float32),
        np.asarray(process_features, dtype=np.float32),
        np.asarray(scan_features, dtype=np.float32),
    ])

def build_graph(obs, env):
    host_vec_size = HostVector.state_size
    obs_2d = obs.reshape(-1, host_vec_size)

    for idx, address in enumerate(env.scenario.address_space):
        host_slice = obs_2d[idx].copy()
        current = read_host_observation(host_slice, address)
        if current["discovered"]:
            if address not in known_host_states:
                known_host_states[address] = current
            else:
                known_host_states[address] = merge_host_observation(
                    known_host_states[address], current
                )

    discovered_hosts = list(known_host_states.keys())
    node_features = [
        host_feature_vector(address, known_host_states[address])
        for address in discovered_hosts
    ]
    host_to_idx = {address: i for i, address in enumerate(discovered_hosts)}

    edge_list = []
    edge_features = []
    for address_a in discovered_hosts:
        for address_b in discovered_hosts:
            if address_a == address_b:
                continue
            subnet_a, subnet_b = address_a[0], address_b[0]
            if env.scenario.topology[subnet_a][subnet_b] != 1:
                continue
            edge_list.append((host_to_idx[address_a], host_to_idx[address_b]))
            reachability = [
                edge_tracker.get((subnet_a, subnet_b, service_idx), {}).get("outcome", 0)
                for service_idx in range(len(SERVICE_VOCAB))
            ]
            edge_features.append(
                reachability + [1 if subnet_a == subnet_b else 0]
            )

    return {
        "discovered_hosts": discovered_hosts,
        "host_to_idx": host_to_idx,
        "node_features": np.asarray(node_features, dtype=np.float32),
        "edges": edge_list,
        "edge_features": np.asarray(edge_features, dtype=np.float32),
    }

def convert_to_PyG(graph):
    if len(graph["node_features"]) > 0:
        x = torch.tensor(graph["node_features"], dtype=torch.float32)
    else:
        x = torch.empty((0, NODE_FEATURE_DIM), dtype=torch.float32)

    if len(graph["edges"]) > 0:
        edge_index = torch.tensor(
            graph["edges"], dtype=torch.long
        ).t().contiguous()
        edge_attr = torch.tensor(
            graph["edge_features"], dtype=torch.float32
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Copied definitions from notebook cell 21 (including markdown cells).
class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=True,
                                edge_dim=edge_dim, dropout=dropout, bias=True)
        self.conv2 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, concat=False,
                                edge_dim=edge_dim, dropout=dropout, bias=True)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x

class NetworkStateEncoder(nn.Module):
    def __init__(self, **gnn_kwargs):
        super().__init__()
        self.gnn = GATEncoder(**gnn_kwargs)

    def forward(self, obs, env):
        graph = build_graph(obs, env)
        pyg_data = convert_to_PyG(graph)
        node_emb = self.gnn(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
        # Mean and max pooling provide both average state and salient-host signals.
        graph_emb = torch.cat([
            node_emb.mean(dim=0),
            node_emb.max(dim=0).values,
        ])
        return graph_emb, node_emb, graph, pyg_data

class ActorNetwork(nn.Module):
    def __init__(self, action_types_per_host, embedding_size=16):
        super().__init__()
        self.actor_layer = nn.Linear(embedding_size, action_types_per_host)

    def forward(self, node_embeddings, mask=None, deterministic=False):
        action_scores = self.actor_layer(node_embeddings).reshape(-1)
        if mask is not None:
            action_scores = action_scores.masked_fill(~mask, -1e9)
        action_distribution = Categorical(logits=action_scores)
        chosen_action = (
            torch.argmax(action_scores)
            if deterministic else action_distribution.sample()
        )
        log_prob = action_distribution.log_prob(chosen_action)
        return chosen_action, log_prob, action_distribution

class Critic(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.critic_layer = nn.Linear(embedding_size, 1)

    def forward(self, graph_emb):
        return self.critic_layer(graph_emb)

# Copied definitions from notebook cell 24 (including markdown cells).
SCAN_ACTION_NAMES = ["service_scan", "os_scan", "subnet_scan", "process_scan"]

def build_action_mask(discovered_hosts, action_types, node_service_state,
                      node_process_state, node_scan_state, known_host_states):
    mask = []
    for host in discovered_hosts:
        remembered = known_host_states[host]
        compromised = bool(remembered["compromised"])
        root_access = remembered["access"] >= AccessLevel.ROOT

        for action_name in action_types:
            if action_name in SCAN_ACTION_NAMES:
                allowed = not node_scan_state.get((host, action_name), False)
                if action_name in ("subnet_scan", "process_scan"):
                    allowed = allowed and compromised
                mask.append(allowed)
            elif action_name in action_type_to_service:
                service_idx = service_to_idx[action_type_to_service[action_name]]
                known = node_service_state.get((host, service_idx), {}).get("known", 0)
                mask.append((not compromised) and known != -1)
            elif action_name in action_type_to_process:
                process_idx = PROCESS_TO_IDX[action_type_to_process[action_name]]
                known = node_process_state.get((host, process_idx), {}).get("known", 0)
                mask.append(compromised and (not root_access) and known != -1)
            else:
                mask.append(False)

    mask = torch.tensor(mask, dtype=torch.bool)
    if not mask.any():
        raise RuntimeError("Action mask contains no valid actions; inspect tracker state.")
    return mask

def build_nasim_action(action_type_name, target, env):
    scan_costs = {
        "service_scan": env.scenario.service_scan_cost,
        "os_scan": env.scenario.os_scan_cost,
        "subnet_scan": env.scenario.subnet_scan_cost,
        "process_scan": env.scenario.process_scan_cost,
    }
    scan_classes = {
        "service_scan": ServiceScan,
        "os_scan": OSScan,
        "subnet_scan": SubnetScan,
        "process_scan": ProcessScan,
    }
    if action_type_name in scan_classes:
        return scan_classes[action_type_name](
            target=target, cost=scan_costs[action_type_name]
        )
    if action_type_name in env.scenario.exploits:
        return Exploit(
            name=action_type_name,
            target=target,
            **env.scenario.exploits[action_type_name],
        )
    if action_type_name in env.scenario.privescs:
        return PrivilegeEscalation(
            name=action_type_name,
            target=target,
            **env.scenario.privescs[action_type_name],
        )
    raise ValueError(f"Unknown action type: {action_type_name}")

# Copied definitions from notebook cell 30 (including markdown cells).
class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.pyg_data_list = []
        self.masks = []

    def store(self, obs, action, log_prob, value, reward, done, pyg_data, mask):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.pyg_data_list.append(pyg_data)
        self.masks.append(mask)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)

def compute_returns_and_advantages(buffer, last_value, gamma=0.99, lam=0.95):
    values = torch.stack(buffer.values).reshape(-1)
    rewards = torch.as_tensor(buffer.rewards, dtype=values.dtype, device=values.device)
    dones = torch.as_tensor(buffer.dones, dtype=values.dtype, device=values.device)
    last_value = torch.as_tensor(last_value, dtype=values.dtype, device=values.device).detach()

    advantages = torch.zeros_like(values)
    last_gae = torch.zeros((), dtype=values.dtype, device=values.device)

    for t in reversed(range(len(buffer))):
        next_value = last_value if t == len(buffer) - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    return returns.detach(), advantages.detach()

def compute_ppo_loss(actor, critic, state_encoder, buffer, returns, advantages,
                     batch_indices, epsilon=0.2, ent_coef=0.01):
    indices = [int(idx) for idx in batch_indices]
    old_log_probs = torch.stack(
        [buffer.log_probs[idx] for idx in indices]
    ).reshape(-1).detach()

    new_log_probs, new_values, entropies = [], [], []
    for idx in indices:
        pyg_data = buffer.pyg_data_list[idx]
        action = buffer.actions[idx]
        mask = buffer.masks[idx]

        node_emb = state_encoder.gnn(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
        graph_emb = torch.cat([
            node_emb.mean(dim=0),
            node_emb.max(dim=0).values,
        ])
        _, _, dist = actor(node_emb, mask)
        new_log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        new_values.append(critic(graph_emb))

    new_log_probs = torch.stack(new_log_probs).reshape(-1)
    entropies = torch.stack(entropies).reshape(-1)
    new_values = torch.stack(new_values).reshape(-1)

    batch_returns = returns[indices]
    batch_advantages = advantages[indices]
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * batch_advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * batch_advantages
    actor_loss = -torch.min(unclipped, clipped).mean()

    critic_loss = F.mse_loss(new_values, batch_returns)
    entropy_bonus = entropies.mean()
    return actor_loss + 0.50 * critic_loss - ent_coef * entropy_bonus

service_to_idx = {name: i for i, name in enumerate(SERVICE_VOCAB)}
PROCESS_TO_IDX = {name: i for i, name in enumerate(PROCESS_VOCAB)}
action_types = SCAN_ACTION_NAMES + ["e_ssh", "e_ftp", "e_http", "pe_tomcat", "pe_daclsvc"]
action_types_per_host = len(action_types)
action_type_to_service = {"e_ssh": "ssh", "e_ftp": "ftp", "e_http": "http"}
action_type_to_process = {"pe_tomcat": "tomcat", "pe_daclsvc": "daclsvc"}

