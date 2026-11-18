"""CLI script for isolated PyTorch model behavioral evaluation.

Invoked inside a disposable worker subprocess under verified OS isolation controls.
Loads candidate weights safely with weights_only=True into a verified architecture class,
executes deterministic probes in eval() mode with torch.no_grad(), and outputs structured
behavioral metrics as JSON.
"""

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from backend.app.modules.model_integrity.architecture.registry import (
    TrustedArchitectureRegistry,
)
from backend.app.modules.model_integrity.behavioral.probes import (
    apply_bounded_perturbation,
    apply_trigger,
    compute_behavioral_fingerprint,
    generate_deterministic_probes,
    get_candidate_triggers,
)


def run_batch_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    batch_size: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs forward inference in bounded batches without gradient tracking.

    Returns:
        tuple of (logits, probabilities, predicted_classes, confidences)
    """
    all_logits: list[torch.Tensor] = []
    n_samples = inputs.shape[0]

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = inputs[start_idx:end_idx]
            out = model(batch)
            all_logits.append(out.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    confs = torch.max(probs, dim=-1).values

    return logits, probs, preds, confs


def compute_logit_digest(row: torch.Tensor) -> str:
    """Computes a stable SHA-256 digest of rounded float logits for canonical serialization."""
    rounded_str = ",".join(f"{float(x):.5f}" for x in row.tolist())
    return hashlib.sha256(rounded_str.encode("utf-8")).hexdigest()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    architecture_id: str,
    registry: TrustedArchitectureRegistry,
) -> torch.nn.Module:
    """Safely loads weights into an authentic trusted architecture instance."""
    raw = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError("Root checkpoint structure is not a dictionary.")

    model = registry.create_instance(architecture_id)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch model behavior and backdoor indicators inside isolated worker."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to candidate checkpoint file.")
    parser.add_argument(
        "--architecture", required=True, help="Identifier of expected trusted architecture."
    )
    parser.add_argument("--probes", type=int, default=30, help="Number of deterministic probes.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for deterministic probe generator."
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Forward pass batch size.")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Sensitivity noise bound.")
    parser.add_argument(
        "--reference-checkpoint",
        default=None,
        help="Optional path to trusted reference model checkpoint.",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Checkpoint file '{checkpoint_path}' does not exist.",
                    "error_type": "FILE_NOT_FOUND",
                }
            )
        )
        sys.exit(0)

    registry = TrustedArchitectureRegistry()
    if not registry.has(args.architecture):
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Architecture '{args.architecture}' is not registered in TrustedArchitectureRegistry.",
                    "error_type": "UNKNOWN_ARCHITECTURE",
                }
            )
        )
        sys.exit(0)

    try:
        model = load_model_from_checkpoint(checkpoint_path, args.architecture, registry)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Failed to safely load checkpoint into architecture '{args.architecture}': {exc}",
                    "error_type": "MODEL_LOAD_ERROR",
                }
            )
        )
        sys.exit(0)

    input_shape = registry.get_default_input_shape(args.architecture)

    try:
        # 1. Generate deterministic clean probes
        probes = generate_deterministic_probes(args.probes, input_shape, seed=args.seed)

        # 2. Baseline inference
        clean_logits, clean_probs, clean_preds, clean_confs = run_batch_inference(
            model, probes, batch_size=args.batch_size
        )

        # Extract penultimate latent representations in batches
        all_features: list[torch.Tensor] = []
        with torch.no_grad():
            for start_idx in range(0, args.probes, args.batch_size):
                end_idx = min(start_idx + args.batch_size, args.probes)
                batch = probes[start_idx:end_idx]
                feats = registry.extract_features(model, args.architecture, batch)
                all_features.append(feats.detach().cpu())

        features = torch.cat(all_features, dim=0)

        # Group latent representations by class ID
        # For synthetic probes, evenly distribute across 10 classes (class_id = i % 10)
        num_classes = 10
        class_latents_map: dict[str, Any] = {}
        for c in range(num_classes):
            class_indices = [i for i in range(args.probes) if (i % num_classes) == c]
            if class_indices:
                class_feats = features[class_indices].numpy()
                class_latents_map[str(c)] = {
                    "shape": list(class_feats.shape),
                    "dtype": str(class_feats.dtype),
                    "data_b64": base64.b64encode(class_feats.tobytes()).decode("ascii"),
                }

        probe_records: list[dict[str, Any]] = []
        for i in range(args.probes):
            probe_records.append(
                {
                    "index": i,
                    "predicted_class": int(clean_preds[i].item()),
                    "confidence": round(float(clean_confs[i].item()), 6),
                    "logit_digest": compute_logit_digest(clean_logits[i]),
                }
            )

        behavioral_fp = compute_behavioral_fingerprint(probe_records)

        # Baseline class distribution
        pred_list = [int(p.item()) for p in clean_preds]
        class_counts: dict[int, int] = {}
        for p in pred_list:
            class_counts[p] = class_counts.get(p, 0) + 1
        clean_mode_count = max(class_counts.values()) if class_counts else 0
        clean_concentration = clean_mode_count / max(1, args.probes)

        # 3. Prediction consistency evaluation (R repeated passes)
        consistency_flips = 0
        max_logit_drift = 0.0

        for _r_idx in range(2):
            rep_logits, _, rep_preds, _ = run_batch_inference(
                model, probes, batch_size=args.batch_size
            )
            diff = (rep_preds != clean_preds).sum().item()
            consistency_flips += int(diff)
            drift = torch.max(torch.abs(rep_logits - clean_logits)).item()
            if drift > max_logit_drift:
                max_logit_drift = drift

        consistency_report = {
            "is_deterministic": consistency_flips == 0 and max_logit_drift <= 1e-4,
            "flip_count": consistency_flips,
            "max_logit_drift": round(max_logit_drift, 6),
            "repetitions": 3,
        }

        # 4. Input sensitivity evaluation (bounded noise in [-epsilon, epsilon])
        pert_probes, deltas = apply_bounded_perturbation(
            probes, epsilon=args.epsilon, seed=args.seed
        )
        pert_logits, pert_probs, pert_preds, pert_confs = run_batch_inference(
            model, pert_probes, batch_size=args.batch_size
        )

        flips = (pert_preds != clean_preds).sum().item()
        flip_rate = flips / max(1, args.probes)
        conf_deltas = torch.abs(pert_confs - clean_confs).mean().item()

        delta_norms = torch.norm(deltas.reshape(args.probes, -1), dim=1) + 1e-8
        logit_diff_norms = torch.norm((pert_logits - clean_logits).reshape(args.probes, -1), dim=1)
        amplification = (logit_diff_norms / delta_norms).mean().item()

        sensitivity_report = {
            "epsilon": args.epsilon,
            "flip_count": flips,
            "flip_rate": round(flip_rate, 4),
            "mean_confidence_delta": round(conf_deltas, 4),
            "mean_logit_amplification": round(amplification, 4),
        }

        # 5. Backdoor indicator evaluation across candidate triggers
        candidate_triggers = get_candidate_triggers(input_shape, trigger_size=4)
        trigger_results: dict[str, Any] = {}
        highest_concentration = 0.0
        most_suspicious_trigger: str | None = None
        target_class_for_suspicious: int | None = None
        max_concentration_delta = 0.0

        for t_name, t_spec in candidate_triggers.items():
            trig_probes = apply_trigger(probes, t_spec)
            _, _, trig_preds, trig_confs = run_batch_inference(
                model, trig_probes, batch_size=args.batch_size
            )

            t_preds_list = [int(p.item()) for p in trig_preds]
            t_counts: dict[int, int] = {}
            for p in t_preds_list:
                t_counts[p] = t_counts.get(p, 0) + 1

            t_mode_class = max(t_counts.keys(), key=lambda k: t_counts[k]) if t_counts else 0
            t_mode_count = t_counts.get(t_mode_class, 0)
            concentration = t_mode_count / max(1, args.probes)
            conc_delta = concentration - clean_concentration

            # Compute average confidence on target class
            target_mask = trig_preds == t_mode_class
            avg_target_conf = (
                trig_confs[target_mask].mean().item() if target_mask.sum() > 0 else 0.0
            )

            trigger_results[t_name] = {
                "target_class": t_mode_class,
                "concentration_ratio": round(concentration, 4),
                "clean_concentration": round(clean_concentration, 4),
                "concentration_delta": round(conc_delta, 4),
                "avg_confidence": round(avg_target_conf, 4),
                "target_sample_count": t_mode_count,
            }

            if concentration > highest_concentration:
                highest_concentration = concentration
                most_suspicious_trigger = t_name
                target_class_for_suspicious = t_mode_class
                max_concentration_delta = conc_delta

        backdoor_report = {
            "evaluated_triggers_count": len(candidate_triggers),
            "highest_concentration": round(highest_concentration, 4),
            "clean_concentration": round(clean_concentration, 4),
            "max_concentration_delta": round(max_concentration_delta, 4),
            "most_suspicious_trigger": most_suspicious_trigger,
            "target_class": target_class_for_suspicious,
            "triggers": trigger_results,
        }

        # 6. Optional trusted reference model comparison
        reference_comparison: dict[str, Any] | None = None
        if args.reference_checkpoint:
            ref_path = Path(args.reference_checkpoint).resolve()
            if ref_path.exists():
                ref_model = load_model_from_checkpoint(ref_path, args.architecture, registry)
                ref_logits, ref_probs, ref_preds, ref_confs = run_batch_inference(
                    ref_model, probes, batch_size=args.batch_size
                )

                ref_records: list[dict[str, Any]] = []
                for i in range(args.probes):
                    ref_records.append(
                        {
                            "index": i,
                            "predicted_class": int(ref_preds[i].item()),
                            "confidence": round(float(ref_confs[i].item()), 6),
                            "logit_digest": compute_logit_digest(ref_logits[i]),
                        }
                    )
                ref_fp = compute_behavioral_fingerprint(ref_records)

                agreed = (clean_preds == ref_preds).sum().item()
                agreement_rate = agreed / max(1, args.probes)
                max_logit_div = torch.max(torch.abs(clean_logits - ref_logits)).item()
                mean_logit_div = torch.mean(torch.abs(clean_logits - ref_logits)).item()
                mean_conf_diff = torch.mean(torch.abs(clean_confs - ref_confs)).item()

                reference_comparison = {
                    "reference_checkpoint": str(ref_path),
                    "reference_behavioral_fingerprint": ref_fp,
                    "top1_agreement_rate": round(agreement_rate, 4),
                    "max_logit_divergence": round(max_logit_div, 6),
                    "mean_logit_divergence": round(mean_logit_div, 6),
                    "mean_confidence_difference": round(mean_conf_diff, 6),
                    "fingerprints_match": behavioral_fp == ref_fp,
                }

        output_payload = {
            "success": True,
            "architecture_id": args.architecture,
            "probes_count": args.probes,
            "probe_seed": args.seed,
            "input_shape": input_shape,
            "behavioral_fingerprint": behavioral_fp,
            "clean_class_distribution": class_counts,
            "clean_concentration": round(clean_concentration, 4),
            "consistency": consistency_report,
            "sensitivity": sensitivity_report,
            "backdoor": backdoor_report,
            "reference_comparison": reference_comparison,
            "sample_records": probe_records[:5],
            "latent_representations": class_latents_map,
        }

        print(json.dumps(output_payload))

    except Exception as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Behavioral probe evaluation failed: {exc}",
                    "error_type": "PROBE_EXECUTION_ERROR",
                }
            )
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
