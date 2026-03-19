import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import model_evaluate
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier

import model_train as mt


def load_model(model_path):
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def compute_best_threshold(y_val, val_pred):
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_pred)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    denom = precisions + recalls
    f1_scores = np.zeros_like(denom)
    valid = denom > 0
    f1_scores[valid] = 2.0 * (precisions[valid] * recalls[valid]) / denom[valid]

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def resolve_threshold(args):
    if args.threshold is not None:
        return float(args.threshold), "user", None

    if args.val_data is not None:
        val_payload = np.load(args.val_data)
        y_val = np.asarray(val_payload["y_val"]).astype(int)
        val_pred = np.asarray(val_payload["val_pred"])
        thr, best_f1 = compute_best_threshold(y_val, val_pred)
        return thr, "val_data_max_f1", best_f1

    return None, "none", None


def naming_tag(campaign_tag, beam_electrons, beam_protons):
    if campaign_tag:
        return campaign_tag
    be = int(beam_electrons) if float(beam_electrons).is_integer() else beam_electrons
    bp = int(beam_protons) if float(beam_protons).is_integer() else beam_protons
    return f"{be}x{bp}"


def apply_tag_if_missing(prefix_path, tag):
    if (tag is None) or (tag == ""):
        return prefix_path
    if tag in prefix_path.name:
        return prefix_path
    return prefix_path.with_name(f"{prefix_path.name}_{tag}")


def build_three_class_masks(x_df, y_true):
    y_true = np.asarray(y_true).astype(int)
    if ("truth_pdg" not in x_df.columns) or ("truth_gen" not in x_df.columns):
        raise ValueError("truth_pdg/truth_gen columns are required for 3-class plots")

    truth_pdg = x_df["truth_pdg"].to_numpy()
    truth_gen = x_df["truth_gen"].to_numpy()
    all_mask = np.ones(len(x_df), dtype=bool)
    scattered_e_mask = y_true == 1
    # Keep the same signed-PDG convention as training labels: scattered electron is pdg == 11.
    hfs_mask = (truth_gen == 1) & (truth_pdg != 11)
    return all_mask, scattered_e_mask, hfs_mask


def shared_bins(values_a, values_b, values_c, feature_name):
    if feature_name == "E_over_p":
        return np.linspace(0.0, 1.5, 51)
    if feature_name == "is_leading_pt":
        return np.array([-0.5, 0.5, 1.5])

    arrays = [arr for arr in (values_a, values_b, values_c) if arr.size > 0]
    if not arrays:
        return np.linspace(0.0, 1.0, 51)

    all_vals = np.concatenate(arrays)
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    return np.linspace(vmin, vmax, 51)


def plot_three_class_feature_distributions(x_df, y_true, output_path):
    all_mask, scattered_e_mask, hfs_mask = build_three_class_masks(x_df, y_true)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feature in enumerate(mt.FEATURE_COLUMNS):
        values = x_df[feature].to_numpy()
        bins = shared_bins(values[all_mask], values[scattered_e_mask], values[hfs_mask], feature)

        axes[i].hist(values[all_mask], bins=bins, histtype="step", color="black", linewidth=1.4, label="all")
        axes[i].hist(
            values[scattered_e_mask],
            bins=bins,
            histtype="step",
            color="tab:red",
            linewidth=1.4,
            label="scattered electron",
        )
        axes[i].hist(
            values[hfs_mask],
            bins=bins,
            histtype="step",
            color="tab:blue",
            linewidth=1.4,
            label="background",
        )

        axes[i].set_title(f"Stress Test: {feature} Distribution")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Number of particles")
        if feature == "is_leading_pt":
            axes[i].set_xlim(-0.5, 1.5)
            axes[i].set_xticks([0, 1])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Stress Test: Input Variable Distributions (all, scattered electron, background)",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_isolation_cone_scan_three_class(events, args, output_path):
    cones = args.isolation_cones
    ncols = 3
    nrows = int(np.ceil(len(cones) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    p_tot, rot_y_angle, rot_x_angle = mt.func.determine_boost(
        args.beam_electrons,
        args.beam_protons,
        args.crossing_angle,
    )

    for ax, cone in zip(axes, cones):
        x_df_cone, y_true_cone, _ = mt.process_features(
            events,
            p_tot,
            rot_y_angle,
            rot_x_angle,
            isolation_cone_size=float(cone),
            signal_generator_statuses=args.signal_generator_statuses,
        )
        all_mask, scattered_e_mask, hfs_mask = build_three_class_masks(x_df_cone, y_true_cone)
        iso = x_df_cone["isolation_frac"].to_numpy()

        ax.hist(iso[all_mask], bins=50, range=(0, 1.05), histtype="step", color="black", label="all")
        ax.hist(
            iso[scattered_e_mask],
            bins=50,
            range=(0, 1.05),
            histtype="step",
            color="tab:red",
            label="scattered electron",
        )
        ax.hist(
            iso[hfs_mask],
            bins=50,
            range=(0, 1.05),
            histtype="step",
            color="tab:blue",
            label="background",
        )
        ax.set_xlabel(r"$E_{calo}^{cand} / \Sigma E_{calo}^{cone}$")
        ax.set_ylabel("counts")
        ax.set_title(f"Stress Test: Isolation Fraction (cone={float(cone):g})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(cones) :]:
        ax.set_visible(False)

    fig.suptitle("Stress Test: Iso Frac for Different Cone Sizes", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run stress-test inference on ROOT datasets")
    parser.add_argument("--model-in", type=str, required=True, help="Path to trained model JSON")
    parser.add_argument("--input-files", type=str, nargs="+", required=True, help="Stress ROOT files")
    parser.add_argument("--output-prefix", type=str, default="model_outputs/stress", help="Prefix for saved outputs")
    parser.add_argument("--campaign-tag", type=str, default=None, help="Optional campaign suffix in output naming")
    parser.add_argument("--beam-electrons", type=float, default=18.0, help="Electron beam energy in GeV")
    parser.add_argument("--beam-protons", type=float, default=275.0, help="Proton beam energy in GeV")
    parser.add_argument("--crossing-angle", type=float, default=-0.025, help="Beam crossing angle in radians")
    parser.add_argument(
        "--isolation-cone-size",
        type=float,
        default=2.5,
        help="Cone size used for the model input feature isolation_frac (default: 2.5)",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Optional fixed score threshold")
    parser.add_argument("--val-data", type=str, default=None, help="Optional *_val.npz to auto-pick threshold by max F1")
    parser.add_argument(
        "--signal-generator-statuses",
        type=int,
        nargs="+",
        default=[1],
        help="Generator-status values considered final-state for selecting the scattered electron (default: 1)",
    )
    parser.add_argument(
        "--isolation-cones",
        type=float,
        nargs="+",
        default=[0.4, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        help="Cone sizes for 3-class isolation scan plots",
    )
    parser.add_argument("--save-csv", action="store_true", help="Also save per-particle scores as CSV")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating evaluation-style plots")
    args = parser.parse_args()

    threshold, threshold_source, best_f1 = resolve_threshold(args)

    out_prefix = Path(args.output_prefix)
    tag = naming_tag(args.campaign_tag, args.beam_electrons, args.beam_protons)
    out_prefix = apply_tag_if_missing(out_prefix, tag)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_in}...")
    model = load_model(args.model_in)

    print(f"Loading stress data from {len(args.input_files)} files...")
    events = mt.load_data(args.input_files)

    print("Building stress features with training-equivalent pipeline...")
    p_tot, rot_y_angle, rot_x_angle = mt.func.determine_boost(
        args.beam_electrons,
        args.beam_protons,
        args.crossing_angle,
    )
    X_df, y_true, truth_has_scattered_event = mt.process_features(
        events,
        p_tot,
        rot_y_angle,
        rot_x_angle,
        isolation_cone_size=args.isolation_cone_size,
        signal_generator_statuses=args.signal_generator_statuses,
    )

    represented_event_ids = np.unique(X_df["event_id"].to_numpy().astype(int))
    n_events_represented = int(represented_event_ids.size)
    truth_flags_represented = truth_has_scattered_event[represented_event_ids]
    n_true_scattered_events_represented = int(np.sum(truth_flags_represented))
    n_events_no_truth_scattered = int(np.sum(~truth_flags_represented))

    print(f"stress events represented: {n_events_represented}")
    print(
        "true scattered electrons in represented stress events "
        "(same first-match logic as labels): "
        f"{n_true_scattered_events_represented}"
    )
    print(
        "represented stress events with no truth scattered electron found: "
        f"{n_events_no_truth_scattered}"
    )

    X = np.asarray(X_df[mt.FEATURE_COLUMNS])
    scores = model.predict_proba(X)[:, 1]

    if (threshold is None) and (len(np.unique(y_true)) == 2):
        threshold, best_f1 = compute_best_threshold(y_true, scores)
        threshold_source = "stress_max_f1"

    metrics = {}
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["average_precision"] = float(average_precision_score(y_true, scores))

    selected_count = None
    selected_fraction = None
    if threshold is not None:
        selected_mask = scores > threshold
        selected_count = int(np.sum(selected_mask))
        selected_fraction = float(np.mean(selected_mask))

    npz_out = out_prefix.with_name(f"{out_prefix.name}_scores.npz")
    np.savez_compressed(
        npz_out,
        feature_names=np.asarray(mt.FEATURE_COLUMNS),
        scores=scores,
        y_true=y_true,
        event_id=X_df["event_id"].to_numpy(),
        Q2=X_df["Q2"].to_numpy(),
        x=X_df["x"].to_numpy(),
        y=X_df["y"].to_numpy(),
        E_over_p=X_df["E_over_p"].to_numpy(),
        isolation_frac=X_df["isolation_frac"].to_numpy(),
        is_leading_pt=X_df["is_leading_pt"].to_numpy(),
        charge=X_df["charge"].to_numpy(),
        pt=X_df["pt"].to_numpy(),
        acoplanarity=X_df["acoplanarity"].to_numpy(),
    )

    if args.save_csv:
        csv_out = out_prefix.with_name(f"{out_prefix.name}_scores.csv")
        out_df = X_df.copy()
        out_df["score"] = scores
        out_df["y_true"] = y_true
        if threshold is not None:
            out_df["pred_is_signal"] = (scores > threshold).astype(int)
        out_df.to_csv(csv_out, index=False)
        print(f"Saved per-particle CSV: {csv_out}")

    summary_out = out_prefix.with_name(f"{out_prefix.name}_summary.txt")
    event_ids = X_df["event_id"].to_numpy()
    n_events = int(np.unique(event_ids).size)
    signal_event_ids = event_ids[y_true == 1]
    n_events_with_true_signal = int(np.unique(signal_event_ids).size)

    signal_counts_per_event = np.bincount(event_ids.astype(int), weights=(y_true == 1).astype(int))
    signal_counts_nonzero = signal_counts_per_event[signal_counts_per_event > 0]
    if signal_counts_nonzero.size > 0:
        signal_per_event_min = int(np.min(signal_counts_nonzero))
        signal_per_event_max = int(np.max(signal_counts_nonzero))
        signal_per_event_mean = float(np.mean(signal_counts_nonzero))
    else:
        signal_per_event_min = 0
        signal_per_event_max = 0
        signal_per_event_mean = 0.0

    with open(summary_out, "w", encoding="ascii") as f:
        f.write(f"model_in={args.model_in}\n")
        f.write(f"n_input_files={len(args.input_files)}\n")
        f.write(f"n_events={n_events}\n")
        f.write(f"n_events_with_truth_scattered_firstmatch={n_true_scattered_events_represented}\n")
        f.write(f"n_events_without_truth_scattered_firstmatch={n_events_no_truth_scattered}\n")
        f.write(f"n_events_with_true_signal={n_events_with_true_signal}\n")
        f.write(f"n_particles={len(scores)}\n")
        f.write(f"n_true_signal={int(np.sum(y_true == 1))}\n")
        f.write(f"n_true_background={int(np.sum(y_true == 0))}\n")
        f.write(f"signal_per_event_min={signal_per_event_min}\n")
        f.write(f"signal_per_event_max={signal_per_event_max}\n")
        f.write(f"signal_per_event_mean={signal_per_event_mean:.6f}\n")
        f.write(f"threshold_source={threshold_source}\n")
        if threshold is not None:
            f.write(f"threshold={threshold:.8f}\n")
            if best_f1 is not None:
                f.write(f"valdata_best_f1={best_f1:.8f}\n")
            f.write(f"selected_count={selected_count}\n")
            f.write(f"selected_fraction={selected_fraction:.8f}\n")
        if "roc_auc" in metrics:
            f.write(f"roc_auc={metrics['roc_auc']:.8f}\n")
            f.write(f"average_precision={metrics['average_precision']:.8f}\n")

    print(f"Saved stress scores NPZ: {npz_out}")
    print(f"Saved stress summary: {summary_out}")
    if threshold is not None:
        print(f"Applied threshold={threshold:.6f} ({threshold_source})")
    else:
        print("No threshold applied (raw score output only).")

    if (not args.no_plots) and (np.unique(y_true).size > 1):
        plot_dir = Path("pictures") / f"stress_{tag}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_prefix = plot_dir / out_prefix.name

        q2 = X_df["Q2"].to_numpy()
        x = X_df["x"].to_numpy()

        # Threshold-based plots require a concrete threshold and a displayable F1 value.
        if threshold is None:
            print("Skipped threshold-dependent plots because no threshold was resolved.")
            return
        if best_f1 is None:
            _, best_f1 = compute_best_threshold(y_true, scores)

        model_evaluate.plot_feature_importance_avg_gain(
            model,
            f"{plot_prefix}_importance_avg_gain.png",
            plot_context="Stress Test",
        )
        model_evaluate.plot_feature_importance_total_gain(
            model,
            f"{plot_prefix}_importance_total_gain.png",
            plot_context="Stress Test",
        )
        model_evaluate.plot_purity_efficiency_curve(
            y_true,
            scores,
            threshold,
            best_f1,
            f"{plot_prefix}_purity_efficiency_bestf1.png",
            plot_context="Stress Test",
        )
        model_evaluate.plot_2d_q2x_maps(
            scores,
            y_true,
            q2,
            x,
            threshold,
            best_f1,
            f"{plot_prefix}_q2x_phase_space.png",
            plot_context="Stress Test",
        )
        model_evaluate.plot_input_distributions_tp(
            X,
            y_true,
            scores,
            threshold,
            f"{plot_prefix}_input_distributions_tp.png",
            plot_context="Stress Test",
        )
        model_evaluate.plot_input_distributions_tn(
            X,
            y_true,
            scores,
            threshold,
            f"{plot_prefix}_input_distributions_tn.png",
            plot_context="Stress Test",
        )
        model_evaluate.plot_input_distributions_fn(
            X,
            y_true,
            scores,
            threshold,
            f"{plot_prefix}_input_distributions_fn.png",
            plot_context="Stress Test",
        )

        if ("truth_pdg" in X_df.columns) and ("truth_gen" in X_df.columns):
            plot_three_class_feature_distributions(
                X_df,
                y_true,
                f"{plot_prefix}_input_distributions_3class.png",
            )
            plot_isolation_cone_scan_three_class(
                events,
                args,
                f"{plot_prefix}_isolation_cone_scan_3class.png",
            )
        else:
            print("Skipped 3-class plots because truth_pdg/truth_gen columns are missing.")

        print(f"Saved stress evaluation plots under {plot_dir}")
    elif args.no_plots:
        print("Skipped plots because --no-plots was set.")
    else:
        print("Skipped plots because stress labels do not contain both classes.")

if __name__ == "__main__":
    main()
