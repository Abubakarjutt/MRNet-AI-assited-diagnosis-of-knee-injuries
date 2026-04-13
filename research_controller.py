import json
import os
import random
import subprocess
from copy import deepcopy
from datetime import datetime


NON_PERSISTED_OVERRIDE_KEYS = {
    "max_train_batches",
    "max_val_batches",
    "save_model",
}


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def merge_missing_defaults(config, defaults):
    merged = deepcopy(defaults)
    merged.update(config or {})
    return merged



def persistent_config_snapshot(config):
    return {
        key: value
        for key, value in merge_missing_defaults(config, {}).items()
        if key not in NON_PERSISTED_OVERRIDE_KEYS
    }



def sanitize_config(config, search_space, default_config):
    sanitized = merge_missing_defaults(config, default_config)
    for key, choices in search_space.items():
        if sanitized.get(key) not in choices:
            sanitized[key] = deepcopy(default_config[key])
    return sanitized



def config_signature(config, search_space, default_config):
    normalized = persistent_config_snapshot(sanitize_config(config, search_space, default_config))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))



def parse_mutations(text):
    return [item.strip() for item in (text or "").split(",") if item.strip()]



def parse_mutation_assignment(token):
    token = token.strip()
    if token.startswith("local:"):
        token = token[len("local:"):]
    if token.startswith("research_prior:"):
        return None
    if ":" not in token or "->" not in token:
        return None
    key, values = token.split(":", 1)
    _, new_value = values.split("->", 1)
    return key.strip(), new_value.strip()



def coerce_like(example, raw_value):
    if isinstance(example, bool):
        return raw_value.lower() in {"1", "true", "yes"}
    if isinstance(example, int) and not isinstance(example, bool):
        try:
            return int(float(raw_value))
        except ValueError:
            return example
    if isinstance(example, float):
        try:
            return float(raw_value)
        except ValueError:
            return example
    return raw_value



def mutation_stats(rows):
    stats = {}
    for row in rows:
        auc = safe_float(row.get("best_val_auc"), 0.0)
        status = row.get("status") or "unknown"
        for token in parse_mutations(row.get("mutations")):
            info = stats.setdefault(
                token,
                {"trials": 0, "keeps": 0, "crashes": 0, "best_auc": 0.0, "auc_sum": 0.0},
            )
            info["trials"] += 1
            info["auc_sum"] += auc
            info["best_auc"] = max(info["best_auc"], auc)
            if status == "keep":
                info["keeps"] += 1
            if status == "crash":
                info["crashes"] += 1
    return stats



def rank_positive_mutations(rows):
    ranked = []
    for token, info in mutation_stats(rows).items():
        if token.startswith("controller:"):
            continue
        if info["keeps"] <= 0 and info["best_auc"] <= 0.0:
            continue
        score = info["keeps"] * 1.5 + info["best_auc"] + (info["auc_sum"] / max(info["trials"], 1))
        ranked.append((score, token, info))
    ranked.sort(reverse=True)
    return ranked



def rank_risky_mutations(rows):
    ranked = []
    for token, info in mutation_stats(rows).items():
        if token.startswith("controller:"):
            continue
        crash_rate = info["crashes"] / max(info["trials"], 1)
        if info["trials"] < 2 or crash_rate < 0.6:
            continue
        ranked.append((crash_rate, token, info))
    ranked.sort(reverse=True)
    return ranked



def crash_rate(rows, lookback=8):
    recent = rows[-lookback:]
    if not recent:
        return 0.0
    crashes = sum(1 for row in recent if row.get("status") == "crash")
    return crashes / len(recent)



def recent_summary(rows, limit=8):
    recent = rows[-limit:]
    out = []
    for row in recent:
        out.append(
            {
                "iteration": row.get("iteration"),
                "candidate_name": row.get("candidate_name"),
                "status": row.get("status"),
                "best_val_auc": row.get("best_val_auc"),
                "mutations": parse_mutations(row.get("mutations")),
            }
        )
    return out



def best_successes(rows, limit=5):
    successful = []
    for row in rows:
        if row.get("status") == "crash":
            continue
        successful.append((safe_float(row.get("best_val_auc"), 0.0), row))
    successful.sort(key=lambda item: item[0], reverse=True)
    result = []
    for auc, row in successful[:limit]:
        result.append(
            {
                "candidate_name": row.get("candidate_name"),
                "best_val_auc": auc,
                "mutations": parse_mutations(row.get("mutations")),
            }
        )
    return result



def controller_context(iteration, parent, result_rows, research_state, search_space, default_config, research_priors):
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iteration": iteration,
        "goal": "Propose the next MRNet architecture experiment that has the best chance of materially improving validation AUC.",
        "parent": {
            "name": parent.get("name"),
            "best_val_auc": parent.get("best_val_auc"),
            "config": sanitize_config(parent.get("config", {}), search_space, default_config),
        },
        "recent_results": recent_summary(result_rows),
        "best_successes": best_successes(result_rows),
        "research_state": research_state,
        "research_priors": research_priors,
        "search_space": search_space,
    }



def bounded_choice(preferred, choices, fallback):
    if preferred in choices:
        return preferred
    return fallback if fallback in choices else choices[0]



def simplify_config(parent_config, search_space, default_config):
    candidate = sanitize_config(parent_config, search_space, default_config)
    candidate["model_type"] = bounded_choice("mobilenet_v3_small", search_space["model_type"], candidate["model_type"])
    candidate["pooling"] = bounded_choice("gem", search_space["pooling"], candidate["pooling"])
    candidate["projection_dim"] = bounded_choice(128, search_space["projection_dim"], candidate["projection_dim"])
    candidate["hidden_dim"] = bounded_choice(256, search_space["hidden_dim"], candidate["hidden_dim"])
    candidate["fusion_depth"] = bounded_choice(2, search_space["fusion_depth"], candidate["fusion_depth"])
    candidate["fusion_gate"] = bounded_choice("none", search_space["fusion_gate"], candidate["fusion_gate"])
    candidate["dropout"] = bounded_choice(0.1, search_space["dropout"], candidate["dropout"])
    candidate["lr"] = bounded_choice(3e-4, search_space["lr"], candidate["lr"])
    candidate["weight_decay"] = bounded_choice(1e-4, search_space["weight_decay"], candidate["weight_decay"])
    candidate["image_size"] = bounded_choice(224, search_space["image_size"], candidate["image_size"])
    return candidate



def choose_underexplored_prior(research_state, research_priors, rng):
    scored = []
    for prior in research_priors:
        stats = research_state.get(prior["name"], {"trials": 0, "keeps": 0})
        score = stats.get("keeps", 0) * 2 - stats.get("trials", 0) + rng.random() * 0.05
        scored.append((score, prior))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]



def apply_prior(base_config, prior):
    candidate = deepcopy(base_config)
    mutations = [f"research_prior:{prior['name']}"]
    for key, value in prior.get("mutations", {}).items():
        old_value = candidate.get(key)
        candidate[key] = value
        if old_value != value:
            mutations.append(f"{key}:{old_value}->{value}")
    return candidate, mutations



def apply_positive_evidence(candidate, rows, search_space, default_config, budget=3):
    chosen = []
    seen_keys = set()
    for _, token, info in rank_positive_mutations(rows):
        parsed = parse_mutation_assignment(token)
        if not parsed:
            continue
        key, raw_value = parsed
        if key in seen_keys or key not in search_space:
            continue
        current = candidate.get(key, default_config[key])
        new_value = coerce_like(default_config[key], raw_value)
        if new_value == current or new_value not in search_space[key]:
            continue
        candidate[key] = new_value
        chosen.append(f"evidence:{token}")
        seen_keys.add(key)
        if len(chosen) >= budget:
            break
    return candidate, chosen



def avoid_risky_choices(candidate, rows, search_space, default_config):
    risky = rank_risky_mutations(rows)
    changes = []
    for _, token, _ in risky[:3]:
        parsed = parse_mutation_assignment(token)
        if not parsed:
            continue
        key, raw_value = parsed
        if key not in search_space:
            continue
        risky_value = coerce_like(default_config[key], raw_value)
        if candidate.get(key) != risky_value:
            continue
        fallback = default_config[key]
        if fallback == risky_value and len(search_space[key]) > 1:
            fallback = search_space[key][0]
        if fallback == risky_value:
            continue
        candidate[key] = fallback
        changes.append(f"risk_avoid:{key}:{risky_value}->{fallback}")
    return candidate, changes



def heuristic_proposal(iteration, parent, result_rows, research_state, search_space, default_config, research_priors, seen_signatures, max_duplicate_retries, rng):
    notes = []
    sources = []
    recent_crash_rate = crash_rate(result_rows)
    if recent_crash_rate >= 0.6:
        candidate = simplify_config(parent.get("config", {}), search_space, default_config)
        mutations = ["controller:stability_reset"]
        notes.append(
            f"Recent crash rate is {recent_crash_rate:.0%}, so the controller is simplifying the search around the strongest stable compact architecture."
        )
    else:
        prior = choose_underexplored_prior(research_state, research_priors, rng)
        sources = [prior.get("source")] if prior.get("source") else []
        candidate, mutations = apply_prior(parent.get("config", {}), prior)
        mutations.insert(0, "controller:heuristic_research_agent")
        notes.append(f"Selected research direction `{prior['name']}` based on underexplored literature-inspired ideas and prior outcomes.")

    candidate = sanitize_config(candidate, search_space, default_config)
    candidate, evidence_mutations = apply_positive_evidence(candidate, result_rows, search_space, default_config)
    candidate, risk_mutations = avoid_risky_choices(candidate, result_rows, search_space, default_config)
    mutations.extend(evidence_mutations)
    mutations.extend(risk_mutations)

    parent_signature = config_signature(parent.get("config", {}), search_space, default_config)
    for attempt in range(max_duplicate_retries):
        signature = config_signature(candidate, search_space, default_config)
        if signature not in seen_signatures and signature != parent_signature:
            return {
                "config": candidate,
                "mutations": mutations,
                "controller_name": "heuristic_research_agent",
                "analysis": notes,
                "hypothesis": "Blend the strongest known compact backbone choices with literature-inspired mutations while avoiding settings associated with recent crashes.",
                "sources": sources,
            }
        key = rng.choice(list(search_space.keys()))
        current = candidate.get(key, default_config[key])
        choices = [choice for choice in search_space[key] if choice != current]
        if not choices:
            continue
        new_value = rng.choice(choices)
        candidate[key] = new_value
        mutations.append(f"controller_retry:{key}:{current}->{new_value}")
        notes.append(f"Adjusted `{key}` on retry {attempt + 1} to avoid a duplicate proposal.")

    return {
        "config": candidate,
        "mutations": mutations,
        "controller_name": "heuristic_research_agent",
        "analysis": notes,
        "hypothesis": "Retry-based exploration around the best known compact architecture.",
        "sources": [],
    }



def call_external_controller(command, context):
    process = subprocess.run(
        command,
        input=json.dumps(context),
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or f"external controller exited with code {process.returncode}")
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("external controller returned invalid JSON") from exc
    if not isinstance(payload, dict) or "config" not in payload:
        raise RuntimeError("external controller response must be a JSON object with a 'config' field")
    return payload



def propose_next_experiment(
    iteration,
    parent,
    result_rows,
    research_state,
    search_space,
    default_config,
    research_priors,
    seen_signatures,
    max_duplicate_retries,
    seed,
    agent_command=None,
):
    rng = random.Random(seed)
    context = controller_context(
        iteration=iteration,
        parent=parent,
        result_rows=result_rows,
        research_state=research_state,
        search_space=search_space,
        default_config=default_config,
        research_priors=research_priors,
    )

    external_error = None
    if agent_command:
        try:
            response = call_external_controller(agent_command, context)
            proposal = {
                "config": sanitize_config(response.get("config", {}), search_space, default_config),
                "mutations": list(response.get("mutations") or ["controller:external_agent"]),
                "controller_name": response.get("controller_name") or "external_research_agent",
                "analysis": list(response.get("analysis") or []),
                "hypothesis": response.get("hypothesis") or "External research agent proposed this candidate.",
                "sources": list(response.get("sources") or []),
            }
            if not any(str(item).startswith("controller:") for item in proposal["mutations"]):
                proposal["mutations"].insert(0, "controller:external_agent")
        except Exception as exc:
            external_error = str(exc)
        else:
            proposal["context"] = context
            return proposal

    proposal = heuristic_proposal(
        iteration=iteration,
        parent=parent,
        result_rows=result_rows,
        research_state=research_state,
        search_space=search_space,
        default_config=default_config,
        research_priors=research_priors,
        seen_signatures=seen_signatures,
        max_duplicate_retries=max_duplicate_retries,
        rng=rng,
    )
    if external_error:
        proposal.setdefault("analysis", []).append(
            f"External research agent was unavailable, so the loop fell back to the built-in controller: {external_error}"
        )
    proposal["context"] = context
    return proposal



def write_research_memo(path, proposal, outcome=None):
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "controller_name": proposal.get("controller_name"),
        "hypothesis": proposal.get("hypothesis"),
        "analysis": proposal.get("analysis") or [],
        "mutations": proposal.get("mutations") or [],
        "config": proposal.get("config") or {},
        "sources": proposal.get("sources") or [],
        "context": proposal.get("context") or {},
    }
    if outcome is not None:
        payload["outcome"] = outcome
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
