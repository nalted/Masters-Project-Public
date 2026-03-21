import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from matplotlib.colors import LogNorm
from sklearn.metrics import average_precision_score, precision_recall_curve


FEATURE_COLUMNS = [
    "E_over_p",
    "isolation_frac",
    "is_leading_pt",
    "charge",
    "pt",
    "acoplanarity",
]

FEATURE_LABELS = [
    "E/p",
    "isolation fraction",
    "leading transverse momentum",
    "charge",
    "transverse momentum",
    "acoplanarity",
]


def load_model(model_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def load_validation_data(npz_path):
    data = np.load(npz_path)
    X_val = np.asarray(data["X_val"])
    y_val = np.asarray(data["y_val"]).astype(int)
    val_pred = np.asarray(data["val_pred"])
    event_Q2 = np.asarray(data["event_Q2"])
    event_x = np.asarray(data["event_x"])
    event_y = np.asarray(data["event_y"])

    # Optional diagnostics arrays (newer training outputs).
    event_id_val = np.asarray(data["event_id_val"]).astype(int) if "event_id_val" in data.files else None
    truth_has_scattered_event_val = (
        np.asarray(data["truth_has_scattered_event_val"]).astype(bool)
        if "truth_has_scattered_event_val" in data.files
        else None
    )
    val_events = np.asarray(data["val_events"]).astype(int) if "val_events" in data.files else None
    all_val_events = np.asarray(data["all_val_events"]).astype(int) if "all_val_events" in data.files else None
    all_val_true_q2 = np.asarray(data["all_val_true_q2"]) if "all_val_true_q2" in data.files else None
    all_val_true_x = np.asarray(data["all_val_true_x"]) if "all_val_true_x" in data.files else None
    all_val_has_scattered = (
        np.asarray(data["all_val_has_scattered"]).astype(bool)
        if "all_val_has_scattered" in data.files
        else None
    )

    return (
        X_val,
        y_val,
        val_pred,
        event_Q2,
        event_x,
        event_y,
        event_id_val,
        truth_has_scattered_event_val,
        val_events,
        all_val_events,
        all_val_true_q2,
        all_val_true_x,
        all_val_has_scattered,
    )


def print_validation_truth_consistency(y_val, event_id_val=None, truth_has_scattered_event_val=None, val_events=None):
    if event_id_val is None:
        print(
            "Validation truth-consistency diagnostics unavailable: "
            "event_id_val missing in validation npz."
        )
        return

    event_id_val = np.asarray(event_id_val).astype(int)
    y_val = np.asarray(y_val).astype(int)

    unique_events = np.unique(event_id_val)
    n_val_events_represented = int(unique_events.size)
    n_val_events_total = int(len(val_events)) if val_events is not None else n_val_events_represented

    val_signal_event_ids = event_id_val[y_val == 1]
    if val_signal_event_ids.size > 0:
        signal_counts_per_event = {
            int(evt): int(cnt)
            for evt, cnt in zip(*np.unique(val_signal_event_ids, return_counts=True))
        }
        n_events_with_signal = len(signal_counts_per_event)
        n_events_with_multiple_signal = int(sum(cnt > 1 for cnt in signal_counts_per_event.values()))
    else:
        n_events_with_signal = 0
        n_events_with_multiple_signal = 0

    n_events_with_zero_signal = int(n_val_events_represented - n_events_with_signal)

    print("validation-event signal-label multiplicity after candidate masking:")
    print(f"events with 0 signal candidate: {n_events_with_zero_signal}")
    print(
        "events with exactly 1 signal candidate: "
        f"{n_val_events_represented - n_events_with_zero_signal - n_events_with_multiple_signal}"
    )
    print(f"events with >1 signal candidate: {n_events_with_multiple_signal}")

    if truth_has_scattered_event_val is not None:
        truth_has_scattered_event_val = np.asarray(truth_has_scattered_event_val).astype(bool)
        if truth_has_scattered_event_val.shape[0] != n_val_events_total:
            print(
                "Validation truth-consistency warning: "
                "truth_has_scattered_event_val length does not match number of validation events."
            )
            return

        n_true_scattered_events = int(np.sum(truth_has_scattered_event_val))
        n_missing_truth = int(np.sum(~truth_has_scattered_event_val))
        print(f"validation events represented after candidate masking: {n_val_events_represented}")
        print(f"validation events total (including filtered-out): {n_val_events_total}")
        print(
            "true scattered electrons in validation events "
            "(same first-match logic as labels): "
            f"{n_true_scattered_events}"
        )
        print(f"validation events with no truth scattered electron found: {n_missing_truth}")
    else:
        print(
            "Validation truth-consistency note: "
            "truth_has_scattered_event_val missing in validation npz."
        )


def _build_event_truth_flags(event_ids, truth_has_scattered_event_val=None, val_events=None, y_val=None):
    unique_event_ids = np.asarray(np.unique(event_ids), dtype=int)

    if truth_has_scattered_event_val is not None and val_events is not None:
        val_events = np.asarray(val_events, dtype=int)
        truth_has_scattered_event_val = np.asarray(truth_has_scattered_event_val, dtype=bool)

        if val_events.shape[0] == truth_has_scattered_event_val.shape[0]:
            truth_lookup = {
                int(evt): bool(flag)
                for evt, flag in zip(val_events, truth_has_scattered_event_val)
            }
            event_truth = np.asarray([truth_lookup.get(int(evt), False) for evt in unique_event_ids], dtype=bool)
            return unique_event_ids, event_truth, "truth_has_scattered_event_val+val_events"

    if y_val is not None:
        # Fallback: event-level truth approximated from surviving particle labels.
        y_val = np.asarray(y_val).astype(int)
        event_truth = np.asarray(
            [np.any(y_val[event_ids == evt] == 1) for evt in unique_event_ids],
            dtype=bool,
        )
        return unique_event_ids, event_truth, "y_val fallback (post-mask truth only)"

    raise ValueError("Unable to build event-level truth flags: missing required arrays")


def build_top1_selected_mask(event_ids, scores, threshold):
    """Select at most one predicted scattered-electron candidate per event.

    For each event, this keeps only the candidate with the maximum score,
    and marks it selected only if that maximum is above ``threshold``.
    Ties are resolved by ``np.argmax`` (first occurrence).
    """
    event_ids = np.asarray(event_ids).astype(int)
    scores = np.asarray(scores, dtype=float)

    if event_ids.shape[0] != scores.shape[0]:
        raise ValueError("Length mismatch between event_ids and scores")

    selected_mask = np.zeros(scores.shape[0], dtype=bool)
    if scores.size == 0:
        return selected_mask

    for evt in np.unique(event_ids):
        idx = np.flatnonzero(event_ids == evt)
        evt_scores = scores[idx]
        best_local = int(np.argmax(evt_scores))
        if evt_scores[best_local] > threshold:
            selected_mask[idx[best_local]] = True

    return selected_mask


def compute_event_level_metrics(
    y_val,
    val_pred,
    event_id_val,
    best_threshold,
    truth_has_scattered_event_val=None,
    val_events=None,
    all_val_events=None,
    all_val_has_scattered=None,
):
    if event_id_val is None:
        return None

    event_id_val = np.asarray(event_id_val).astype(int)
    y_val = np.asarray(y_val).astype(int)
    val_pred = np.asarray(val_pred, dtype=float)

    represented_event_ids = np.asarray(np.unique(event_id_val), dtype=int)
    selected_mask = build_top1_selected_mask(event_id_val, val_pred, best_threshold)
    event_pred_positive_lookup = {
        int(evt): bool(np.any(selected_mask[event_id_val == evt]))
        for evt in represented_event_ids
    }

    if (all_val_events is not None) and (all_val_has_scattered is not None):
        all_val_events = np.asarray(all_val_events, dtype=int)
        all_val_has_scattered = np.asarray(all_val_has_scattered, dtype=bool)
        event_ids_for_metrics = all_val_events
        event_truth = all_val_has_scattered
        truth_source = "all_val_has_scattered (unfiltered validation events)"
    else:
        event_ids_for_metrics, event_truth, truth_source = _build_event_truth_flags(
            event_id_val,
            truth_has_scattered_event_val=truth_has_scattered_event_val,
            val_events=val_events,
            y_val=y_val,
        )

    event_pred_positive = np.asarray(
        [event_pred_positive_lookup.get(int(evt), False) for evt in event_ids_for_metrics],
        dtype=bool,
    )

    tp = int(np.sum(event_truth & event_pred_positive))
    fp = int(np.sum((~event_truth) & event_pred_positive))
    fn = int(np.sum(event_truth & (~event_pred_positive)))
    tn = int(np.sum((~event_truth) & (~event_pred_positive)))

    event_efficiency = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    event_purity = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "n_events_total": int(len(event_ids_for_metrics)),
        "n_truth_scattered_events": int(np.sum(event_truth)),
        "n_predicted_scattered_events": int(np.sum(event_pred_positive)),
        "n_events_predicted_none": int(np.sum(~event_pred_positive)),
        "tp_events": tp,
        "fp_events": fp,
        "fn_events": fn,
        "tn_events": tn,
        "event_efficiency": float(event_efficiency),
        "event_purity": float(event_purity),
        "truth_source": truth_source,
    }


def print_event_level_metrics(metrics, threshold):
    if metrics is None:
        print("Event-level metrics unavailable: event_id_val missing in validation npz.")
        return

    print("\nEvent-level metrics (per event, thresholded candidate scores):")
    print(f"threshold: {threshold:.4f}")
    print(f"truth source: {metrics['truth_source']}")
    print(f"validation events used: {metrics['n_events_total']}")
    print(f"truth scattered events: {metrics['n_truth_scattered_events']}")
    print(f"events predicted as having scattered electron (>=1 candidate): {metrics['n_predicted_scattered_events']}")
    print(f"events predicted with no scattered electron candidate: {metrics['n_events_predicted_none']}")
    print(
        "event efficiency = TP_event / (TP_event + FN_event): "
        f"{metrics['event_efficiency']:.4f}"
    )
    print(
        "event purity = TP_event / (TP_event + FP_event): "
        f"{metrics['event_purity']:.4f}"
    )
    print(
        "event confusion counts: "
        f"TP={metrics['tp_events']}, FP={metrics['fp_events']}, "
        f"FN={metrics['fn_events']}, TN={metrics['tn_events']}"
    )


def compute_best_threshold(y_val, val_pred):
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_pred)
    if thresholds.size == 0:
        # Degenerate case (e.g. constant scores): keep a neutral threshold.
        return 0.5, 0.0

    precisions = precisions[:-1]
    recalls = recalls[:-1]

    denom = precisions + recalls
    f1_scores = np.zeros_like(denom)
    valid = denom > 0
    f1_scores[valid] = 2.0 * (precisions[valid] * recalls[valid]) / denom[valid]

    max_f1 = float(np.max(f1_scores))
    # If several thresholds give the same best F1, pick the largest threshold
    # to avoid always falling back to tiny/zero cut values.
    tied_best = np.flatnonzero(np.isclose(f1_scores, max_f1, rtol=1e-12, atol=1e-12))
    best_idx = int(tied_best[-1]) if tied_best.size else int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return best_threshold, best_f1


def build_event_level_pr_arrays(
    event_id_val,
    val_pred,
    y_val,
    truth_has_scattered_event_val=None,
    val_events=None,
    all_val_events=None,
    all_val_has_scattered=None,
):
    if event_id_val is None:
        raise ValueError("event_id_val is required to build event-level PR arrays")

    event_id_val = np.asarray(event_id_val, dtype=int)
    val_pred = np.asarray(val_pred, dtype=float)
    y_val = np.asarray(y_val).astype(int)

    represented_event_ids = np.asarray(np.unique(event_id_val), dtype=int)
    event_score_lookup = {
        int(evt): float(np.max(val_pred[event_id_val == evt]))
        for evt in represented_event_ids
    }

    if (all_val_events is not None) and (all_val_has_scattered is not None):
        all_val_events = np.asarray(all_val_events, dtype=int)
        all_val_has_scattered = np.asarray(all_val_has_scattered, dtype=int)
        event_scores = np.asarray(
            [event_score_lookup.get(int(evt), 0.0) for evt in all_val_events],
            dtype=float,
        )
        return all_val_has_scattered, event_scores, "all_val_has_scattered (unfiltered validation events)"

    event_ids_for_metrics, event_truth, truth_source = _build_event_truth_flags(
        event_id_val,
        truth_has_scattered_event_val=truth_has_scattered_event_val,
        val_events=val_events,
        y_val=y_val,
    )
    event_scores = np.asarray(
        [event_score_lookup.get(int(evt), 0.0) for evt in event_ids_for_metrics],
        dtype=float,
    )
    return event_truth.astype(int), event_scores, truth_source


def _plot_importance(model, output_path, importance_type, title, plot_context="Training"):
    score = model.get_booster().get_score(importance_type=importance_type)

    values = []
    for i, label in enumerate(FEATURE_LABELS):
        values.append(score.get(f"f{i}", 0.0))

    order = np.argsort(values)
    sorted_labels = [FEATURE_LABELS[i] for i in order]
    sorted_values = np.asarray(values)[order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(sorted_labels, sorted_values, color="tab:blue", alpha=0.8)
    ax.set_xlabel(importance_type)
    ax.set_title(f"[Candidate-Level] {title} ({plot_context})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_avg_gain(model, output_path, plot_context="Training"):
    _plot_importance(
        model,
        output_path,
        importance_type="gain",
        title="Feature Importance (Average Gain)",
        plot_context=plot_context,
    )


def plot_feature_importance_total_gain(model, output_path, plot_context="Training"):
    _plot_importance(
        model,
        output_path,
        importance_type="total_gain",
        title="Feature Importance (Total Gain)",
        plot_context=plot_context,
    )


def plot_purity_efficiency_curve(y_val, val_pred, best_threshold, best_f1, output_path, plot_context="Training"):
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_pred)
    ap = average_precision_score(y_val, val_pred)

    best_thresh_idx = int(np.argmin(np.abs(thresholds - best_threshold)))
    best_precision = precisions[best_thresh_idx]
    best_recall = recalls[best_thresh_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls, precisions, color="tab:blue", lw=1.5, label=f"Purity-Efficiency curve (AP={ap:.3f})")
    ax.scatter(
        best_recall,
        best_precision,
        color="tab:red",
        s=50,
        zorder=5,
        label=(
            f"max F1={best_f1:.3f} @ threshold={best_threshold:.3f}\n"
            f"purity={best_precision:.3f}\n"
            f"efficiency={best_recall:.3f}"
        ),
    )
    ax.axhline(best_precision, color="tab:red", ls="--", alpha=0.3)
    ax.axvline(best_recall, color="tab:red", ls="--", alpha=0.3)
    ax.set_xlabel("efficiency")
    ax.set_ylabel("purity")
    ax.set_title(f"[{plot_context}] Purity-Efficiency Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig.suptitle(
        f"[{plot_context}] Purity-Efficiency (Threshold = {best_threshold:.3f}, F1 = {best_f1:.3f})",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def efficiency_purity_2d_q2x(
    val_pred,
    y_true,
    event_q2,
    event_x,
    threshold=0.5,
    q2_bins=None,
    x_bins=None,
    min_denom=1,
):
    if q2_bins is None:
        q2_bins = np.geomspace(1.0, 100.0, 10)
    if x_bins is None:
        x_bins = np.geomspace(1e-4, 1.0, 20)

    q2_vals = np.asarray(event_q2, dtype=float)
    x_vals = np.asarray(event_x, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(val_pred) > threshold

    n_q2 = len(q2_bins) - 1
    n_x = len(x_bins) - 1

    eff_map = np.full((n_q2, n_x), np.nan, dtype=float)
    pur_map = np.full((n_q2, n_x), np.nan, dtype=float)
    n_sig_map = np.zeros((n_q2, n_x), dtype=int)
    n_pred_map = np.zeros((n_q2, n_x), dtype=int)
    n_tot_map = np.zeros((n_q2, n_x), dtype=int)

    for iq in range(n_q2):
        q2_lo, q2_hi = q2_bins[iq], q2_bins[iq + 1]
        q2_mask = (q2_vals >= q2_lo) & (q2_vals < q2_hi)

        for ix in range(n_x):
            x_lo, x_hi = x_bins[ix], x_bins[ix + 1]
            in_bin = q2_mask & (x_vals >= x_lo) & (x_vals < x_hi)

            if not np.any(in_bin):
                continue

            yb = y_true[in_bin]
            pb = y_pred[in_bin]

            tp = np.sum((yb == 1) & (pb == 1))
            fp = np.sum((yb == 0) & (pb == 1))
            fn = np.sum((yb == 1) & (pb == 0))

            n_sig = tp + fn
            n_pred = tp + fp

            n_sig_map[iq, ix] = n_sig
            n_pred_map[iq, ix] = n_pred
            n_tot_map[iq, ix] = int(np.sum(in_bin))

            if n_sig >= min_denom:
                eff_map[iq, ix] = tp / n_sig
            if n_pred >= min_denom:
                pur_map[iq, ix] = tp / n_pred

    return {
        "q2_bins": q2_bins,
        "x_bins": x_bins,
        "efficiency": eff_map,
        "purity": pur_map,
        "n_signal": n_sig_map,
        "n_predicted_signal": n_pred_map,
        "n_total": n_tot_map,
    }


def efficiency_purity_2d_q2x_event_level(
    val_pred,
    event_id_val,
    all_val_events,
    all_val_true_q2,
    all_val_true_x,
    all_val_has_scattered,
    threshold=0.5,
    q2_bins=None,
    x_bins=None,
    min_denom=1,
):
    if event_id_val is None:
        raise ValueError("event_id_val is required for event-level Q2-x maps")
    if any(v is None for v in [all_val_events, all_val_true_q2, all_val_true_x, all_val_has_scattered]):
        raise ValueError("all_val_* arrays are required for total event-level Q2-x maps")

    if q2_bins is None:
        q2_bins = np.geomspace(1.0, 100.0, 10)
    if x_bins is None:
        x_bins = np.geomspace(1e-4, 1.0, 20)

    event_id_val = np.asarray(event_id_val).astype(int)
    all_val_events = np.asarray(all_val_events, dtype=int)
    all_val_true_q2 = np.asarray(all_val_true_q2, dtype=float)
    all_val_true_x = np.asarray(all_val_true_x, dtype=float)
    all_val_has_scattered = np.asarray(all_val_has_scattered, dtype=bool)
    val_pred = np.asarray(val_pred, dtype=float)

    # Build event-level prediction lookup from top-1 candidate selections.
    represented_event_ids = np.asarray(np.unique(event_id_val), dtype=int)
    selected_mask = build_top1_selected_mask(event_id_val, val_pred, threshold)
    pred_positive_lookup = {
        int(evt): bool(np.any(selected_mask[event_id_val == evt]))
        for evt in represented_event_ids
    }

    # Prediction status for all validation events (including those with no kept candidates).
    all_val_pred_positive = np.asarray(
        [pred_positive_lookup.get(int(evt), False) for evt in all_val_events],
        dtype=bool,
    )

    n_q2 = len(q2_bins) - 1
    n_x = len(x_bins) - 1

    eff_map = np.full((n_q2, n_x), np.nan, dtype=float)
    pur_map = np.full((n_q2, n_x), np.nan, dtype=float)
    n_truth_map = np.zeros((n_q2, n_x), dtype=int)
    n_pred_map = np.zeros((n_q2, n_x), dtype=int)
    n_tot_map = np.zeros((n_q2, n_x), dtype=int)

    for iq in range(n_q2):
        q2_lo, q2_hi = q2_bins[iq], q2_bins[iq + 1]
        q2_mask = (all_val_true_q2 >= q2_lo) & (all_val_true_q2 < q2_hi)

        for ix in range(n_x):
            x_lo, x_hi = x_bins[ix], x_bins[ix + 1]
            in_bin_kin = q2_mask & (all_val_true_x >= x_lo) & (all_val_true_x < x_hi)

            n_total_in_bin = int(np.sum(in_bin_kin))
            if n_total_in_bin <= 0:
                continue

            in_bin_truth = in_bin_kin & all_val_has_scattered
            in_bin_background = in_bin_kin & (~all_val_has_scattered)

            n_truth_in_bin = int(np.sum(in_bin_truth))
            tp_count = int(np.sum(all_val_pred_positive & in_bin_truth))
            fp_count = int(np.sum(all_val_pred_positive & in_bin_background))
            n_pred_in_bin = tp_count + fp_count

            n_truth_map[iq, ix] = n_truth_in_bin
            n_pred_map[iq, ix] = n_pred_in_bin
            n_tot_map[iq, ix] = n_total_in_bin

            if n_truth_in_bin >= min_denom:
                eff_map[iq, ix] = tp_count / n_truth_in_bin
            if n_pred_in_bin >= min_denom:
                pur_map[iq, ix] = tp_count / n_pred_in_bin

    return {
        "q2_bins": q2_bins,
        "x_bins": x_bins,
        "efficiency": eff_map,
        "purity": pur_map,
        "n_truth": n_truth_map,
        "n_predicted_signal": n_pred_map,
        "n_total": n_tot_map,
        "truth_source": "all_val_has_scattered (unfiltered validation events)",
    }


def plot_2d_q2x_maps(
    val_pred,
    y_val,
    event_q2,
    event_x,
    best_threshold,
    best_f1,
    output_path,
    plot_context="Training",
):
    q2x = efficiency_purity_2d_q2x(
        val_pred=val_pred,
        y_true=y_val,
        event_q2=event_q2,
        event_x=event_x,
        threshold=best_threshold,
        q2_bins=np.geomspace(1.0, 100.0, 10),
        x_bins=np.geomspace(1e-4, 1.0, 20),
        min_denom=1,
    )

    q2_bins = q2x["q2_bins"]
    x_bins = q2x["x_bins"]
    eff_2d = q2x["efficiency"]
    pur_2d = q2x["purity"]
    n_tot_2d = q2x["n_total"]

    q2_centers = np.sqrt(q2_bins[:-1] * q2_bins[1:])
    x_centers = np.sqrt(x_bins[:-1] * x_bins[1:])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 10), constrained_layout=True)

    m1 = ax1.pcolormesh(x_bins, q2_bins, eff_2d, shading="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Q2 (GeV^2)")
    ax1.set_title("[Candidate-Level] Efficiency")
    cb1 = fig.colorbar(m1, ax=ax1)
    cb1.set_label("efficiency")

    m2 = ax2.pcolormesh(x_bins, q2_bins, pur_2d, shading="auto", vmin=0.0, vmax=1.0, cmap="magma")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Q2 (GeV^2)")
    ax2.set_title("[Candidate-Level] Purity")
    cb2 = fig.colorbar(m2, ax=ax2)
    cb2.set_label("purity")

    n_display = np.where(n_tot_2d > 0, n_tot_2d, np.nan)
    vmax = np.nanmax(n_display) if np.any(np.isfinite(n_display)) else 1.0
    m3 = ax3.pcolormesh(
        x_bins,
        q2_bins,
        n_display,
        shading="auto",
        cmap="cividis",
        norm=LogNorm(vmin=1, vmax=max(vmax, 1.0)),
    )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("x")
    ax3.set_ylabel("Q2 (GeV^2)")
    ax3.set_title("[Candidate-Level] Bin Density (N per Bin)")
    cb3 = fig.colorbar(m3, ax=ax3)
    cb3.set_label("count per bin (log scale)")

    for iq, q2c in enumerate(q2_centers):
        for ix, xc in enumerate(x_centers):
            if n_tot_2d[iq, ix] <= 0:
                continue

            eff_val = eff_2d[iq, ix]
            pur_val = pur_2d[iq, ix]

            if np.isfinite(eff_val):
                ax1.text(
                    xc,
                    q2c,
                    f"{eff_val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if eff_val < 0.55 else "black",
                )

            if np.isfinite(pur_val):
                ax2.text(
                    xc,
                    q2c,
                    f"{pur_val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if pur_val < 0.55 else "black",
                )

    fig.suptitle(
        (
            "[Candidate-Level] Efficiency and Purity in Q2-x "
            f"(Threshold = {best_threshold:.3f}, Candidate F1 = {best_f1:.3f})"
        ),
        fontsize=14,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_2d_q2x_maps_event_level(
    val_pred,
    y_val,
    event_id_val,
    event_q2,
    event_x,
    best_threshold,
    best_f1,
    output_path,
    plot_context="Training",
    truth_has_scattered_event_val=None,
    val_events=None,
    all_val_events=None,
    all_val_true_q2=None,
    all_val_true_x=None,
    all_val_has_scattered=None,
):
    using_total_unfiltered = all(v is not None for v in [all_val_events, all_val_true_q2, all_val_true_x, all_val_has_scattered])
    if using_total_unfiltered:
        q2x = efficiency_purity_2d_q2x_event_level(
            val_pred=val_pred,
            event_id_val=event_id_val,
            all_val_events=all_val_events,
            all_val_true_q2=all_val_true_q2,
            all_val_true_x=all_val_true_x,
            all_val_has_scattered=all_val_has_scattered,
            threshold=best_threshold,
            q2_bins=np.geomspace(1.0, 100.0, 10),
            x_bins=np.geomspace(1e-4, 1.0, 20),
            min_denom=1,
        )
    else:
        # Backward compatibility for older val npz without all_val_* arrays.
        q2x = efficiency_purity_2d_q2x(
            val_pred=val_pred,
            y_true=y_val,
            event_q2=event_q2,
            event_x=event_x,
            threshold=best_threshold,
            q2_bins=np.geomspace(1.0, 100.0, 10),
            x_bins=np.geomspace(1e-4, 1.0, 20),
            min_denom=1,
        )
        q2x["truth_source"] = "fallback candidate-level bins (all_val_* unavailable)"

    q2_bins = q2x["q2_bins"]
    x_bins = q2x["x_bins"]
    eff_2d = q2x["efficiency"]
    n_tot_2d = q2x["n_total"]
    truth_source = q2x["truth_source"]

    q2_centers = np.sqrt(q2_bins[:-1] * q2_bins[1:])
    x_centers = np.sqrt(x_bins[:-1] * x_bins[1:])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)

    m1 = ax1.pcolormesh(x_bins, q2_bins, eff_2d, shading="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Q2 (GeV^2)")
    ax1.set_title("[Event-Level] Total Event Efficiency")
    cb1 = fig.colorbar(m1, ax=ax1)
    cb1.set_label("efficiency")

    n_display = np.where(n_tot_2d > 0, n_tot_2d, np.nan)
    vmax = np.nanmax(n_display) if np.any(np.isfinite(n_display)) else 1.0
    m2 = ax2.pcolormesh(
        x_bins,
        q2_bins,
        n_display,
        shading="auto",
        cmap="cividis",
        norm=LogNorm(vmin=1, vmax=max(vmax, 1.0)),
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Q2 (GeV^2)")
    ax2.set_title("[Event-Level] Event Density (N events per bin)")
    cb2 = fig.colorbar(m2, ax=ax2)
    cb2.set_label("event count per bin (log scale)")

    for iq, q2c in enumerate(q2_centers):
        for ix, xc in enumerate(x_centers):
            if n_tot_2d[iq, ix] <= 0:
                continue

            eff_val = eff_2d[iq, ix]
            if np.isfinite(eff_val):
                ax1.text(
                    xc,
                    q2c,
                    f"{eff_val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if eff_val < 0.55 else "black",
                )

    fig.suptitle(
        (
            "[Event-Level] Total Efficiency in Q2-x "
            f"(Threshold = {best_threshold:.3f}, Event F1 = {best_f1:.3f}, truth = {truth_source})"
        ),
        fontsize=14,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _shared_bins(values_a, values_b, feature_name):
    plot_ranges = {"E_over_p": (0.0, 1.5)}
    if feature_name == "is_leading_pt":
        return np.array([-0.5, 0.5, 1.5])
    if feature_name in plot_ranges:
        lo, hi = plot_ranges[feature_name]
        return np.linspace(lo, hi, 51)

    arrays = []
    if values_a.size > 0:
        arrays.append(values_a)
    if values_b.size > 0:
        arrays.append(values_b)

    if not arrays:
        return np.linspace(0.0, 1.0, 51)

    all_vals = np.concatenate(arrays)
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    return np.linspace(vmin, vmax, 51)


def plot_input_distributions_tp(
    X_val,
    y_val,
    val_pred,
    best_threshold,
    output_path,
    plot_context="Training",
    event_id_val=None,
):
    if event_id_val is not None:
        predicted_signal_mask = build_top1_selected_mask(event_id_val, val_pred, best_threshold)
    else:
        predicted_signal_mask = val_pred > best_threshold
    true_signal_mask = y_val == 1

    X_pred_signal = X_val[predicted_signal_mask]
    X_true_signal = X_val[true_signal_mask]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURE_COLUMNS):
        bins = _shared_bins(X_true_signal[:, i], X_pred_signal[:, i], feature)

        axes[i].hist(
            X_true_signal[:, i],
            bins=bins,
            histtype="step",
            color="tab:red",
            linewidth=1.5,
            alpha=0.6,
            label="True Scattered Electron",
        )
        axes[i].hist(
            X_pred_signal[:, i],
            bins=bins,
            histtype="step",
            color="tab:blue",
            linewidth=1.5,
            label="Predicted Scattered Electron",
        )

        axes[i].set_title(f"[Candidate-Level] TP: {feature} (threshold > {best_threshold:.2f})")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Number of particles")
        if feature == "is_leading_pt":
            axes[i].set_xlim(-0.5, 1.5)
            axes[i].set_xticks([0, 1])
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(
        f"[Candidate-Level] Input Distributions (TP) ({plot_context}, Threshold = {best_threshold:.3f})",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_input_distributions_tn(
    X_val,
    y_val,
    val_pred,
    best_threshold,
    output_path,
    plot_context="Training",
    event_id_val=None,
):
    true_background_mask = y_val == 0
    if event_id_val is not None:
        predicted_background_mask = ~build_top1_selected_mask(event_id_val, val_pred, best_threshold)
    else:
        predicted_background_mask = val_pred <= best_threshold

    X_true_background = X_val[true_background_mask]
    X_pred_background = X_val[predicted_background_mask]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURE_COLUMNS):
        bins = _shared_bins(X_true_background[:, i], X_pred_background[:, i], feature)

        axes[i].hist(
            X_true_background[:, i],
            bins=bins,
            histtype="step",
            color="tab:red",
            linewidth=1.5,
            alpha=0.7,
            label="True Background",
        )
        axes[i].hist(
            X_pred_background[:, i],
            bins=bins,
            histtype="step",
            color="tab:blue",
            linewidth=1.5,
            label="Predicted Background",
        )

        axes[i].set_title(f"[Candidate-Level] TN: {feature} (threshold <= {best_threshold:.2f})")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Number of particles")
        if feature == "is_leading_pt":
            axes[i].set_xlim(-0.5, 1.5)
            axes[i].set_xticks([0, 1])
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(
        f"[Candidate-Level] Input Distributions (TN) ({plot_context}, Threshold = {best_threshold:.3f})",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_input_distributions_fn(
    X_val,
    y_val,
    val_pred,
    best_threshold,
    output_path,
    plot_context="Training",
    event_id_val=None,
):
    true_signal_mask = y_val == 1
    if event_id_val is not None:
        predicted_background_mask = ~build_top1_selected_mask(event_id_val, val_pred, best_threshold)
    else:
        predicted_background_mask = val_pred <= best_threshold
    missed_signal_mask = true_signal_mask & predicted_background_mask

    X_true_signal = X_val[true_signal_mask]
    X_missed_signal = X_val[missed_signal_mask]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURE_COLUMNS):
        bins = _shared_bins(X_true_signal[:, i], X_missed_signal[:, i], feature)

        axes[i].hist(
            X_true_signal[:, i],
            bins=bins,
            histtype="step",
            color="tab:green",
            linewidth=1.5,
            alpha=0.7,
            label="All True Scattered Electrons",
        )
        axes[i].hist(
            X_missed_signal[:, i],
            bins=bins,
            histtype="step",
            color="tab:purple",
            linewidth=1.5,
            label="Missed Scattered Electrons (FN)",
        )

        axes[i].set_title(f"[Candidate-Level] FN: {feature} (threshold <= {best_threshold:.2f})")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Number of particles")
        if feature == "is_leading_pt":
            axes[i].set_xlim(-0.5, 1.5)
            axes[i].set_xticks([0, 1])
        axes[i].set_yscale("log")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(
        f"[Candidate-Level] Input Distributions (FN) ({plot_context}, Threshold = {best_threshold:.3f})",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_input_distributions_three_class(X_df, y_true, output_path, plot_context="Training"):
    required = {"truth_pdg", "truth_gen"}
    missing = [col for col in required if col not in X_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for 3-class plot: {missing}")

    y_true = np.asarray(y_true).astype(int)

    is_all = np.ones(len(X_df), dtype=bool)
    is_scattered_e = y_true == 1
    # Keep the same signed-PDG convention as training: scattered electron is pdg == 11.
    is_hfs = (X_df["truth_gen"].to_numpy() == 1) & (X_df["truth_pdg"].to_numpy() != 11)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURE_COLUMNS):
        values = X_df[feature].to_numpy()

        bins = _shared_bins(values[is_scattered_e], values[is_hfs], feature)

        axes[i].hist(
            values[is_all],
            bins=bins,
            histtype="step",
            color="black",
            linewidth=1.4,
            label="all",
        )
        axes[i].hist(
            values[is_scattered_e],
            bins=bins,
            histtype="step",
            color="tab:red",
            linewidth=1.4,
            label="scattered electron",
        )
        axes[i].hist(
            values[is_hfs],
            bins=bins,
            histtype="step",
            color="tab:blue",
            linewidth=1.4,
            label="background",
        )

        axes[i].set_title(f"[Candidate-Level] {feature} Distribution")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Number of particles")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(
        f"[Candidate-Level] Input Distributions (all, scattered electron, background) ({plot_context})",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained XGBoost electron-ID model")
    parser.add_argument("--model-in", type=str, required=True, help="Path to trained model JSON")
    parser.add_argument("--val-data", type=str, required=True, help="Path to *_val.npz from training")
    parser.add_argument("--output-prefix", type=str, default="pictures/eval", help="Output file prefix")
    parser.add_argument("--campaign-tag", type=str, default=None, help="Optional campaign tag for output naming")
    parser.add_argument("--beam-electrons", type=float, default=18.0, help="Electron beam energy in GeV")
    parser.add_argument("--beam-protons", type=float, default=275.0, help="Proton beam energy in GeV")
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix)
    if args.campaign_tag:
        tag = args.campaign_tag
    else:
        be = int(args.beam_electrons) if float(args.beam_electrons).is_integer() else args.beam_electrons
        bp = int(args.beam_protons) if float(args.beam_protons).is_integer() else args.beam_protons
        tag = f"{be}x{bp}"

    tagged_name = output_prefix.name if tag in output_prefix.name else f"{output_prefix.name}_{tag}"
    plot_dir = Path("pictures") / f"eval_{tag}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = plot_dir / tagged_name

    print(f"Loading model from {args.model_in}...")
    model = load_model(args.model_in)

    print(f"Loading validation arrays from {args.val_data}...")
    (
        X_val,
        y_val,
        val_pred,
        event_Q2,
        event_x,
        event_y,
        event_id_val,
        truth_has_scattered_event_val,
        val_events,
        all_val_events,
        all_val_true_q2,
        all_val_true_x,
        all_val_has_scattered,
    ) = load_validation_data(args.val_data)

    if len(y_val) != len(val_pred):
        raise ValueError("Length mismatch between y_val and val_pred in validation npz")

    best_threshold, best_f1 = compute_best_threshold(y_val, val_pred)
    print(f"[Candidate-Level] Maximum F1 Score: {best_f1:.4f}")
    print(f"[Candidate-Level] Optimal Threshold: {best_threshold:.4f}")

    print_validation_truth_consistency(
        y_val,
        event_id_val=event_id_val,
        truth_has_scattered_event_val=truth_has_scattered_event_val,
        val_events=val_events,
    )

    event_metrics = compute_event_level_metrics(
        y_val,
        val_pred,
        event_id_val,
        best_threshold,
        truth_has_scattered_event_val=truth_has_scattered_event_val,
        val_events=val_events,
        all_val_events=all_val_events,
        all_val_has_scattered=all_val_has_scattered,
    )
    print_event_level_metrics(event_metrics, best_threshold)

    plot_feature_importance_avg_gain(
        model,
        f"{output_prefix}_importance_avg_gain_candidate_level.png",
        plot_context="Training",
    )
    plot_feature_importance_total_gain(
        model,
        f"{output_prefix}_importance_total_gain_candidate_level.png",
        plot_context="Training",
    )
    plot_purity_efficiency_curve(
        y_val,
        val_pred,
        best_threshold,
        best_f1,
        f"{output_prefix}_purity_efficiency_bestf1_candidate_level.png",
        plot_context="Candidate-Level",
    )
    plot_2d_q2x_maps(
        val_pred,
        y_val,
        event_Q2,
        event_x,
        best_threshold,
        best_f1,
        f"{output_prefix}_q2x_phase_space_candidate_level.png",
        plot_context="Candidate-Level",
    )
    plot_2d_q2x_maps_event_level(
        val_pred,
        y_val,
        event_id_val,
        event_Q2,
        event_x,
        best_threshold,
        best_f1,
        f"{output_prefix}_q2x_phase_space_event_level.png",
        plot_context="Event-Level",
        truth_has_scattered_event_val=truth_has_scattered_event_val,
        val_events=val_events,
        all_val_events=all_val_events,
        all_val_true_q2=all_val_true_q2,
        all_val_true_x=all_val_true_x,
        all_val_has_scattered=all_val_has_scattered,
    )
    plot_input_distributions_tp(
        X_val,
        y_val,
        val_pred,
        best_threshold,
        f"{output_prefix}_input_distributions_tp_candidate_level.png",
        plot_context="Training",
        event_id_val=event_id_val,
    )
    plot_input_distributions_tn(
        X_val,
        y_val,
        val_pred,
        best_threshold,
        f"{output_prefix}_input_distributions_tn_candidate_level.png",
        plot_context="Training",
        event_id_val=event_id_val,
    )
    plot_input_distributions_fn(
        X_val,
        y_val,
        val_pred,
        best_threshold,
        f"{output_prefix}_input_distributions_fn_candidate_level.png",
        plot_context="Training",
        event_id_val=event_id_val,
    )

    print("Saved evaluation plots.")


if __name__ == "__main__":
    main()
