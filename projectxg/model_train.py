import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import awkward as ak
import uproot
from xgboost import XGBClassifier
import vector
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import functions as func

FEATURE_COLUMNS = [
    "E_over_p",
    "isolation_frac",
    "is_leading_pt",
    "charge",
    "pt",
    "acoplanarity",
]

FILTER_NAME = [
    "ReconstructedChargedParticles/ReconstructedChargedParticles.*",
    "MCParticles/MCParticles.*",
    "EcalBarrelClusters/EcalBarrelClusters.*",
    "EcalEndcapPClusters/EcalEndcapPClusters.*",
    "EcalEndcapNClusters/EcalEndcapNClusters.*",
    "EcalBarrelClusterAssociations/EcalBarrelClusterAssociations.*",
    "EcalEndcapPClusterAssociations/EcalEndcapPClusterAssociations.*",
    "EcalEndcapNClusterAssociations/EcalEndcapNClusterAssociations.*",
    "ReconstructedChargedParticleAssociations/ReconstructedChargedParticleAssociations.*",
    "InclusiveKinematicsTruth/InclusiveKinematicsTruth.*",
    "_ReconstructedChargedParticleAssociations_*",
    "_EcalBarrelClusterAssociations_*",
    "_EcalEndcapPClusterAssociations_*",
    "_EcalEndcapNClusterAssociations_*",
]


def _pick_field(events, candidates):
    for name in candidates:
        if name in events.fields:
            return events[name]
    raise KeyError(f"None of the expected branches found: {candidates}")


def load_data(file_paths):
    trees = [uproot.open(file)["events"] for file in file_paths]
    events_list = [tree.arrays(filter_name=FILTER_NAME, library="ak") for tree in trees]
    events = ak.concatenate(events_list, axis=0)
    return events

def _cone_label(cone_size):
    return f"{cone_size:g}".replace("-", "m").replace(".", "p")


def process_features(
    events,
    p_tot,
    rot_y_angle,
    rot_x_angle,
    isolation_cone_size=2.5,
    isolation_cone_sizes=None,
    signal_generator_statuses=None,
):
    vector.register_awkward()
    if signal_generator_statuses is None:
        signal_generator_statuses = [1]
    signal_generator_statuses = {int(s) for s in signal_generator_statuses}

    mc_pdg = events["MCParticles.PDG"]
    mc_gen = events["MCParticles.generatorStatus"]

    trueQ2 = events["InclusiveKinematicsTruth.Q2"]
    truex = events["InclusiveKinematicsTruth.x"]
    truey = events["InclusiveKinematicsTruth.y"]

    px_raw = events["ReconstructedChargedParticles.momentum.x"]
    py_raw = events["ReconstructedChargedParticles.momentum.y"]
    pz_raw = events["ReconstructedChargedParticles.momentum.z"]
    charge = events["ReconstructedChargedParticles.charge"]

    mass = 0.000511
    massless_E = np.sqrt(px_raw**2 + py_raw**2 + pz_raw**2)

    barrel_E = events["EcalBarrelClusters.energy"]
    endcapP_E = events["EcalEndcapPClusters.energy"]
    endcapN_E = events["EcalEndcapNClusters.energy"]

    barrel_pos_x = events["EcalBarrelClusters.position.x"]
    barrel_pos_y = events["EcalBarrelClusters.position.y"]
    barrel_pos_z = events["EcalBarrelClusters.position.z"]
    endcapP_pos_x = events["EcalEndcapPClusters.position.x"]
    endcapP_pos_y = events["EcalEndcapPClusters.position.y"]
    endcapP_pos_z = events["EcalEndcapPClusters.position.z"]
    endcapN_pos_x = events["EcalEndcapNClusters.position.x"]
    endcapN_pos_y = events["EcalEndcapNClusters.position.y"]
    endcapN_pos_z = events["EcalEndcapNClusters.position.z"]

    reco_vec = vector.Array(
        ak.zip(
            {
                "px": px_raw,
                "py": py_raw,
                "pz": pz_raw,
                "energy": np.sqrt(px_raw**2 + py_raw**2 + pz_raw**2 + mass**2),
            }
        )
    )
    pt = reco_vec.pt
    p = np.sqrt(px_raw**2 + py_raw**2 + pz_raw**2)

    reco_to_mc_rec = _pick_field(
        events,
        [
            "ReconstructedChargedParticleAssociations.recID",
            "_ReconstructedChargedParticleAssociations_rec.index",
        ],
    )
    reco_to_mc_sim = _pick_field(
        events,
        [
            "ReconstructedChargedParticleAssociations.simID",
            "_ReconstructedChargedParticleAssociations_sim.index",
        ],
    )

    barrel_clu_idx = _pick_field(
        events,
        [
            "EcalBarrelClusterAssociations.recID",
            "_EcalBarrelClusterAssociations_rec.index",
        ],
    )
    barrel_mc_idx = _pick_field(
        events,
        [
            "EcalBarrelClusterAssociations.simID",
            "_EcalBarrelClusterAssociations_sim.index",
        ],
    )
    endcapN_clu_idx = _pick_field(
        events,
        [
            "EcalEndcapNClusterAssociations.recID",
            "_EcalEndcapNClusterAssociations_rec.index",
        ],
    )
    endcapN_mc_idx = _pick_field(
        events,
        [
            "EcalEndcapNClusterAssociations.simID",
            "_EcalEndcapNClusterAssociations_sim.index",
        ],
    )
    endcapP_clu_idx = _pick_field(
        events,
        [
            "EcalEndcapPClusterAssociations.recID",
            "_EcalEndcapPClusterAssociations_rec.index",
        ],
    )
    endcapP_mc_idx = _pick_field(
        events,
        [
            "EcalEndcapPClusterAssociations.simID",
            "_EcalEndcapPClusterAssociations_sim.index",
        ],
    )

    boosted_px, boosted_py, boosted_pz, boosted_E = func.apply_boost(
        px_raw,
        py_raw,
        pz_raw,
        massless_E,
        p_tot,
        rot_y_angle,
        rot_x_angle,
    )

    results_vectorised = [
        func.match_via_mc_vectorised(
            r_rec,
            r_sim,
            bcl,
            bmc,
            bE,
            pcl,
            pmc,
            pE,
            ncl,
            nmc,
            nE,
            nreco,
        )
        for r_rec, r_sim, bcl, bmc, bE, pcl, pmc, pE, ncl, nmc, nE, nreco in zip(
            reco_to_mc_rec,
            reco_to_mc_sim,
            barrel_clu_idx,
            barrel_mc_idx,
            barrel_E,
            endcapP_clu_idx,
            endcapP_mc_idx,
            endcapP_E,
            endcapN_clu_idx,
            endcapN_mc_idx,
            endcapN_E,
            ak.num(px_raw),
        )
    ]

    barrel_E_per_reco = ak.Array([r[0] for r in results_vectorised])
    endcapN_E_per_reco = ak.Array([r[1] for r in results_vectorised])
    endcapP_E_per_reco = ak.Array([r[2] for r in results_vectorised])

    matched_calo_E = ak.where(
        barrel_E_per_reco > 0,
        barrel_E_per_reco,
        ak.where(
            endcapN_E_per_reco > 0,
            endcapN_E_per_reco,
            ak.where(endcapP_E_per_reco > 0, endcapP_E_per_reco, 0.0),
        ),
    )

    matched_E_over_p = ak.where(
        (p > 0) & (charge != 0),
        matched_calo_E / p,
        np.nan,
    )

    barrel_eta_clu, barrel_phi_clu = func.xyz_to_eta_phi(barrel_pos_x, barrel_pos_y, barrel_pos_z)
    endcapP_eta_clu, endcapP_phi_clu = func.xyz_to_eta_phi(endcapP_pos_x, endcapP_pos_y, endcapP_pos_z)
    endcapN_eta_clu, endcapN_phi_clu = func.xyz_to_eta_phi(endcapN_pos_x, endcapN_pos_y, endcapN_pos_z)

    all_calo_eta = ak.concatenate([barrel_eta_clu, endcapP_eta_clu, endcapN_eta_clu], axis=1)
    all_calo_phi = ak.concatenate([barrel_phi_clu, endcapP_phi_clu, endcapN_phi_clu], axis=1)
    all_calo_E = ak.concatenate([barrel_E, endcapP_E, endcapN_E], axis=1)

    results_eta = [
        func.match_via_mc_vectorised(
            r_rec,
            r_sim,
            bcl,
            bmc,
            bV,
            pcl,
            pmc,
            pV,
            ncl,
            nmc,
            nV,
            nreco,
        )
        for r_rec, r_sim, bcl, bmc, bV, pcl, pmc, pV, ncl, nmc, nV, nreco in zip(
            reco_to_mc_rec,
            reco_to_mc_sim,
            barrel_clu_idx,
            barrel_mc_idx,
            barrel_eta_clu,
            endcapP_clu_idx,
            endcapP_mc_idx,
            endcapP_eta_clu,
            endcapN_clu_idx,
            endcapN_mc_idx,
            endcapN_eta_clu,
            ak.num(px_raw),
        )
    ]
    barrel_eta_per_part = ak.Array([r[0] for r in results_eta])
    endcapN_eta_per_part = ak.Array([r[1] for r in results_eta])
    endcapP_eta_per_part = ak.Array([r[2] for r in results_eta])

    results_phi = [
        func.match_via_mc_vectorised(
            r_rec,
            r_sim,
            bcl,
            bmc,
            bV,
            pcl,
            pmc,
            pV,
            ncl,
            nmc,
            nV,
            nreco,
        )
        for r_rec, r_sim, bcl, bmc, bV, pcl, pmc, pV, ncl, nmc, nV, nreco in zip(
            reco_to_mc_rec,
            reco_to_mc_sim,
            barrel_clu_idx,
            barrel_mc_idx,
            barrel_phi_clu,
            endcapP_clu_idx,
            endcapP_mc_idx,
            endcapP_phi_clu,
            endcapN_clu_idx,
            endcapN_mc_idx,
            endcapN_phi_clu,
            ak.num(px_raw),
        )
    ]
    barrel_phi_per_part = ak.Array([r[0] for r in results_phi])
    endcapN_phi_per_part = ak.Array([r[1] for r in results_phi])
    endcapP_phi_per_part = ak.Array([r[2] for r in results_phi])

    matched_calo_eta = ak.where(
        barrel_E_per_reco > 0,
        barrel_eta_per_part,
        ak.where(
            endcapN_E_per_reco > 0,
            endcapN_eta_per_part,
            ak.where(endcapP_E_per_reco > 0, endcapP_eta_per_part, 0.0),
        ),
    )
    matched_calo_phi = ak.where(
        barrel_E_per_reco > 0,
        barrel_phi_per_part,
        ak.where(
            endcapN_E_per_reco > 0,
            endcapN_phi_per_part,
            ak.where(endcapP_E_per_reco > 0, endcapP_phi_per_part, 0.0),
        ),
    )

    if isolation_cone_sizes is None:
        isolation_cone_sizes = [isolation_cone_size]
    # Preserve user order while removing duplicates.
    ordered_unique_cones = list(dict.fromkeys(float(c) for c in isolation_cone_sizes))
    if float(isolation_cone_size) not in ordered_unique_cones:
        ordered_unique_cones.append(float(isolation_cone_size))

    isolation_frac_by_cone = {}
    for cone_size in ordered_unique_cones:
        iso_calo_total = func.calculate_isolation_vectorised(
            matched_calo_eta,
            matched_calo_phi,
            all_calo_eta,
            all_calo_phi,
            all_calo_E,
            cone_size=cone_size,
        )
        isolation_frac_by_cone[cone_size] = ak.where(
            iso_calo_total > 0,
            matched_calo_E / iso_calo_total,
            0.0,
        )

    # Track whether each raw event has a truth scattered electron according to
    # the same first-match logic used for labels.
    truth_has_scattered_event = []
    for pdgs, gens in zip(mc_pdg, mc_gen):
        found = False
        for pdg, gen in zip(pdgs, gens):
            if int(pdg) == 11 and int(gen) in signal_generator_statuses:
                found = True
                break
        truth_has_scattered_event.append(found)
    truth_has_scattered_event = np.asarray(truth_has_scattered_event, dtype=bool)

    labels = []
    truth_pdg_per_reco = []
    truth_gen_per_reco = []
    for rec_ids, sim_ids, pdgs, gens, n in zip(
        reco_to_mc_rec,
        reco_to_mc_sim,
        mc_pdg,
        mc_gen,
        ak.num(boosted_px),
    ):
        target_mc = -1
        for idx, (pdg, gen) in enumerate(zip(pdgs, gens)):
            if int(pdg) == 11 and int(gen) in signal_generator_statuses:
                target_mc = idx
                break

        mc_for_reco = [None] * n
        for rec_id, sim_id in zip(rec_ids, sim_ids):
            ri = int(rec_id)
            if 0 <= ri < n:
                mc_for_reco[ri] = int(sim_id)

        lab = [1 if mc_for_reco[i] == target_mc and target_mc >= 0 else 0 for i in range(n)]
        pdg_lab = [0] * n
        gen_lab = [0] * n
        for i in range(n):
            sim_idx = mc_for_reco[i]
            if sim_idx is not None and 0 <= sim_idx < len(pdgs):
                pdg_lab[i] = int(pdgs[sim_idx])
                gen_lab[i] = int(gens[sim_idx])

        labels.append(lab)
        truth_pdg_per_reco.append(pdg_lab)
        truth_gen_per_reco.append(gen_lab)

    labels = ak.Array(labels)
    truth_pdg = ak.Array(truth_pdg_per_reco)
    truth_gen = ak.Array(truth_gen_per_reco)

    # Candidate definition requires matched calorimeter energy.
    # Diagnostics below compare before/after against the full
    # ReconstructedChargedParticles branch population.
    calo_cluster_mask = matched_calo_E > 0

    labels_flat = ak.to_numpy(ak.flatten(labels)).astype(int)
    calo_mask_flat = ak.to_numpy(ak.flatten(calo_cluster_mask)).astype(bool)

    n_scattered_total = int(np.sum(labels_flat == 1))
    n_background_total = int(np.sum(labels_flat == 0))
    n_scattered_kept_calo = int(np.sum((labels_flat == 1) & calo_mask_flat))
    n_background_kept_calo = int(np.sum((labels_flat == 0) & calo_mask_flat))
    mask = calo_cluster_mask

    boosted_px_kept = boosted_px[mask]
    boosted_py_kept = boosted_py[mask]
    pt_kept = pt[mask]
    charge_kept = charge[mask]
    e_over_p_kept = matched_E_over_p[mask]
    iso_primary_kept = isolation_frac_by_cone[float(isolation_cone_size)][mask]
    truth_pdg_kept = truth_pdg[mask]
    truth_gen_kept = truth_gen[mask]
    labels_kept = labels[mask]

    # Recompute event-dependent features on the kept-particle view so labels/features are self-consistent.
    acoplanarity_kept = func.calc_acoplanarity(
        boosted_px_kept,
        boosted_py_kept,
        boosted_px_kept,
        boosted_py_kept,
    )
    is_leading_pt_kept = func.find_greatest_pt(pt_kept)

    Q2_per_event = ak.fill_none(ak.firsts(trueQ2), 0.0)
    x_per_event = ak.fill_none(ak.firsts(truex), 0.0)
    y_per_event = ak.fill_none(ak.firsts(truey), 0.0)

    Q2_per_particle = ak.flatten(ak.broadcast_arrays(Q2_per_event, boosted_px_kept)[0])
    x_per_particle = ak.flatten(ak.broadcast_arrays(x_per_event, boosted_px_kept)[0])
    y_per_particle = ak.flatten(ak.broadcast_arrays(y_per_event, boosted_px_kept)[0])
    event_index = ak.flatten(ak.broadcast_arrays(ak.local_index(boosted_px_kept, axis=0), boosted_px_kept)[0])

    X_df = pd.DataFrame({
        "E_over_p": ak.to_numpy(ak.flatten(e_over_p_kept)),
        "isolation_frac": ak.to_numpy(ak.flatten(iso_primary_kept)),
        "is_leading_pt": ak.to_numpy(ak.flatten(is_leading_pt_kept)).astype(int),
        "charge": ak.to_numpy(ak.flatten(charge_kept)),
        "pt": ak.to_numpy(ak.flatten(pt_kept)),
        "acoplanarity": ak.to_numpy(ak.flatten(acoplanarity_kept)),
        "truth_pdg": ak.to_numpy(ak.flatten(truth_pdg_kept)).astype(int),
        "truth_gen": ak.to_numpy(ak.flatten(truth_gen_kept)).astype(int),
        "Q2": ak.to_numpy(Q2_per_particle),
        "x": ak.to_numpy(x_per_particle),
        "y": ak.to_numpy(y_per_particle),
        "event_id": ak.to_numpy(event_index),
    })

    for cone_size in ordered_unique_cones:
        col_name = f"isolation_frac_cone_{_cone_label(cone_size)}"
        X_df[col_name] = ak.to_numpy(ak.flatten(isolation_frac_by_cone[cone_size][mask]))

    y = ak.to_numpy(ak.flatten(labels_kept))

    efficiency_counts = {
        "n_events_total": int(len(truth_has_scattered_event)),
        "n_truth_scattered_events": int(np.sum(truth_has_scattered_event)),
        "n_signal_candidates_pre_calo": n_scattered_total,
        "n_signal_candidates_post_calo": n_scattered_kept_calo,
        "n_background_candidates_pre_calo": n_background_total,
        "n_background_candidates_post_calo": n_background_kept_calo,
    }

    return X_df, y, truth_has_scattered_event, efficiency_counts


def print_and_save_efficiency_chain(
    efficiency_counts,
    n_train_events,
    n_val_events,
    n_train_signal,
    n_val_signal,
    output_path,
):
    """Print and save the end-to-end selection-efficiency chain."""
    n_events_total = int(efficiency_counts["n_events_total"])
    n_truth_scattered_events = int(efficiency_counts["n_truth_scattered_events"])
    n_signal_pre_calo = int(efficiency_counts["n_signal_candidates_pre_calo"])
    n_signal_post_calo = int(efficiency_counts["n_signal_candidates_post_calo"])
    n_background_pre_calo = int(efficiency_counts["n_background_candidates_pre_calo"])
    n_background_post_calo = int(efficiency_counts["n_background_candidates_post_calo"])

    n_used_signal = int(n_train_signal + n_val_signal)
    n_used_events = int(n_train_events + n_val_events)
    n_lost_truth_to_label = int(n_truth_scattered_events - n_signal_pre_calo)
    n_lost_label_to_calo = int(n_signal_pre_calo - n_signal_post_calo)
    n_split_diff = int(n_signal_post_calo - n_used_signal)
    n_background_lost_calo = int(n_background_pre_calo - n_background_post_calo)

    eff_truth_to_label = (100.0 * n_signal_pre_calo / n_truth_scattered_events) if n_truth_scattered_events > 0 else 0.0
    eff_label_to_calo = (100.0 * n_signal_post_calo / n_signal_pre_calo) if n_signal_pre_calo > 0 else 0.0
    eff_truth_to_final = (100.0 * n_used_signal / n_truth_scattered_events) if n_truth_scattered_events > 0 else 0.0
    signal_loss_pct = (100.0 * n_lost_label_to_calo / n_signal_pre_calo) if n_signal_pre_calo > 0 else 0.0
    background_loss_pct = (100.0 * n_background_lost_calo / n_background_pre_calo) if n_background_pre_calo > 0 else 0.0

    chain_lines = [
        "=" * 70,
        "SELECTION EFFICIENCY CHAIN",
        "=" * 70,
        f"All events loaded: {n_events_total}",
        f"Truth-scattered events (event-level): {n_truth_scattered_events}",
        "",
        "Signal-candidate chain:",
        f"1) Reco signal candidates before calo mask: {n_signal_pre_calo}",
        f"2) Reco signal candidates after calo mask:  {n_signal_post_calo}",
        f"   - loss at calo mask: {n_lost_label_to_calo} ({(100.0 - eff_label_to_calo):.2f}%)",
        f"3) Signal candidates used in train+val:      {n_used_signal}",
        f"   - split consistency (step2-step3): {n_split_diff}",
        "",
        "Class-dependent calo-mask candidate losses:",
        f"- scattered electron: {n_lost_label_to_calo}/{n_signal_pre_calo} ({signal_loss_pct:.2f}%)",
        f"- background:         {n_background_lost_calo}/{n_background_pre_calo} ({background_loss_pct:.2f}%)",
        "",
        "Derived efficiencies:",
        f"- truth event -> reco signal candidate (step1 / truth): {eff_truth_to_label:.2f}%",
        f"- reco signal -> calo-matched signal (step2 / step1):   {eff_label_to_calo:.2f}%",
        f"- truth event -> final used signal (step3 / truth):     {eff_truth_to_final:.2f}%",
        "",
        "Cross-check notes:",
        f"- truth-to-label difference (truth - step1): {n_lost_truth_to_label}",
        f"- events used in train+val after candidate masking: {n_used_events}",
        "=" * 70,
    ]

    chain_text = "\n".join(chain_lines)
    print("\n" + chain_text + "\n")

    with open(output_path, "w") as f:
        f.write(chain_text + "\n")


def train_model(X_df, y, truth_has_scattered_event=None):
    unique_events = X_df["event_id"].unique()
    if truth_has_scattered_event is not None:
        # Split on all raw events so validation can include events that produced no kept candidates.
        all_events = np.arange(len(truth_has_scattered_event), dtype=int)
    else:
        all_events = np.asarray(unique_events, dtype=int)

    train_events, val_events = train_test_split(all_events, test_size=0.2, random_state=42)

    train_mask = X_df["event_id"].isin(train_events).to_numpy()
    val_mask = X_df["event_id"].isin(val_events).to_numpy()

    if truth_has_scattered_event is not None:
        val_event_indices = np.asarray(val_events, dtype=int)
        val_truth_flags = truth_has_scattered_event[val_event_indices]
    else:
        val_truth_flags = None

    X = np.asarray(X_df[FEATURE_COLUMNS])
    y = np.asarray(y)

    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    event_info_df = X_df[["Q2", "x", "y", "event_id"]]
    event_info_val = event_info_df[val_mask].reset_index(drop=True)

    n_background = int(np.sum(y_train == 0))
    n_signal = int(np.sum(y_train == 1))
    scale_pos_weight = (n_background / n_signal) if n_signal > 0 else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        eta=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=1.0,
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        use_label_encoder=False,
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    val_pred = model.predict_proba(X_val)[:, 1]

    return (
        model,
        X_train,
        X_val,
        y_train,
        y_val,
        event_info_val,
        val_pred,
        len(train_events),
        len(val_events),
        np.asarray(val_events, dtype=int),
        val_truth_flags,
    )


def naming_tag(campaign_tag, beam_electrons, beam_protons):
    if campaign_tag:
        return campaign_tag
    be = int(beam_electrons) if float(beam_electrons).is_integer() else beam_electrons
    bp = int(beam_protons) if float(beam_protons).is_integer() else beam_protons
    return f"{be}x{bp}"


def print_and_save_training_summary(
    n_train_events,
    n_val_events,
    n_train_particles,
    n_val_particles,
    n_train_signal,
    n_val_signal,
    output_path,
):
    """Print and save a training summary to file."""
    summary_lines = [
        "" * 70,
        "TRAINING SUMMARY",
        "=" * 70,
        f"Training set:   {n_train_events:6d} events, {n_train_particles:8d} candidate particles",
        f"  └─ Signal (label=1): {n_train_signal:d}",
        "",
        f"Validation set: {n_val_events:6d} events, {n_val_particles:8d} candidate particles",
        f"  └─ Signal (label=1): {n_val_signal:d}",
        "=" * 70,
    ]
    
    summary_text = "\n".join(summary_lines)
    
    with open(output_path, "w") as f:
        f.write(summary_text + "\n")


def _shared_bins(all_vals, scat_vals, hfs_vals, feature_name):
    arrays = [
        np.asarray(all_vals, dtype=float),
        np.asarray(scat_vals, dtype=float),
        np.asarray(hfs_vals, dtype=float),
    ]
    finite_parts = [a[np.isfinite(a)] for a in arrays if a.size > 0]
    if not finite_parts:
        return 40

    merged = np.concatenate(finite_parts)
    if merged.size == 0:
        return 40

    lo = float(np.min(merged))
    hi = float(np.max(merged))
    if lo == hi:
        eps = 1e-6 if lo == 0.0 else abs(lo) * 1e-3
        lo -= eps
        hi += eps

    if feature_name == "is_leading_pt":
        return np.array([-0.5, 0.5, 1.5])
    if feature_name == "E_over_p":
        return np.linspace(0.0, 1.5, 60)
    if feature_name == "pt":
        return np.linspace(0.0, 5.0, 60)

    return np.linspace(lo, hi, 50)


def _class_masks(X_df, y_true):
    y_true = np.asarray(y_true).astype(int)

    is_all = np.ones(len(X_df), dtype=bool)
    is_scattered_e = y_true == 1

    if ("truth_pdg" in X_df.columns) and ("truth_gen" in X_df.columns):
        # Keep the same signed-PDG convention as label building: scattered electron is pdg == 11.
        is_hfs = (X_df["truth_gen"].to_numpy() == 1) & (X_df["truth_pdg"].to_numpy() != 11)
    else:
        is_hfs = np.zeros(len(X_df), dtype=bool)

    return is_all, is_scattered_e, is_hfs


def _plot_three_class_hist(ax, values, bins, is_all, is_scattered_e, is_hfs):
    ax.hist(
        values[is_all],
        bins=bins,
        histtype="step",
        color="black",
        linewidth=1.4,
        label="all",
    )
    ax.hist(
        values[is_scattered_e],
        bins=bins,
        histtype="step",
        color="tab:red",
        linewidth=1.4,
        label="scattered electron",
    )
    ax.hist(
        values[is_hfs],
        bins=bins,
        histtype="step",
        color="tab:blue",
        linewidth=1.4,
        label="background",
    )


def plot_e_over_p_distribution(X_df, y_true, output_path):
    is_all, is_scattered_e, is_hfs = _class_masks(X_df, y_true)
    values = X_df["E_over_p"].to_numpy()
    bins = np.linspace(0.0, 1.5, 60)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    _plot_three_class_hist(ax, values, bins, is_all, is_scattered_e, is_hfs)
    ax.set_xlim(0.0, 1.5)
    ax.set_title("Training: E/p Input Variable Distribution")
    ax.set_xlabel("E/p")
    ax.set_ylabel("Number of particles")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_isolation_distributions(X_df, y_true, cone_sizes, output_path):
    is_all, is_scattered_e, is_hfs = _class_masks(X_df, y_true)
    cone_sizes = [float(c) for c in cone_sizes]

    n_cols = min(3, max(1, len(cone_sizes)))
    n_rows = int(np.ceil(len(cone_sizes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, cone_size in enumerate(cone_sizes):
        col_name = f"isolation_frac_cone_{_cone_label(cone_size)}"
        if col_name not in X_df.columns:
            continue

        values = X_df[col_name].to_numpy()
        bins = _shared_bins(values[is_all], values[is_scattered_e], values[is_hfs], col_name)
        ax = axes_flat[i]
        _plot_three_class_hist(ax, values, bins, is_all, is_scattered_e, is_hfs)
        ax.set_title(f"Training: Isolation Fraction (cone={cone_size:g})")
        ax.set_xlabel("isolation fraction")
        ax.set_ylabel("Number of particles")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    for j in range(len(cone_sizes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Training: Iso Frac for Different Cone Sizes", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_acoplanarity_distribution(X_df, y_true, output_path):
    is_all, is_scattered_e, is_hfs = _class_masks(X_df, y_true)
    values = X_df["acoplanarity"].to_numpy()
    bins = _shared_bins(values[is_all], values[is_scattered_e], values[is_hfs], "acoplanarity")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    _plot_three_class_hist(ax, values, bins, is_all, is_scattered_e, is_hfs)
    ax.set_title("Training: Acoplanarity Input Variable Distribution")
    ax.set_xlabel("acoplanarity")
    ax.set_ylabel("Number of particles")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_leading_pt_pt_charge_distributions(X_df, y_true, output_path):
    is_all, is_scattered_e, is_hfs = _class_masks(X_df, y_true)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for i, feature in enumerate(["is_leading_pt", "pt", "charge"]):
        values = X_df[feature].to_numpy()
        bins = _shared_bins(values[is_all], values[is_scattered_e], values[is_hfs], feature)

        _plot_three_class_hist(axes[i], values, bins, is_all, is_scattered_e, is_hfs)
        axes[i].set_title(f"Training: {feature} Input Variable Distribution")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Number of particles")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

        if feature == "pt":
            axes[i].set_xlim(0.0, 3.0)

    fig.suptitle("Training: Input Variable Distributions (is_leading_pt, pt, charge)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost electron ID model")
    parser.add_argument('--model-out', type=str, required=True,
                       help='Output filename for trained model (e.g., xgb_18x275_28jan2026.json)')
    parser.add_argument('--input-files', type=str, nargs='+', required=True,
                       help='Input ROOT files for training (e.g., *.root or file1.root file2.root ...)')
    parser.add_argument('--beam-electrons', type=float, default=18.0,
                       help='Electron beam energy in GeV (default: 18)')
    parser.add_argument('--beam-protons', type=float, default=275.0,
                       help='Proton beam energy in GeV (default: 275)')
    parser.add_argument('--crossing-angle', type=float, default=-0.025,
                       help='Beam crossing angle in radians (default: -0.025)')
    parser.add_argument('--campaign-tag', type=str, default=None,
                       help='Optional campaign tag; otherwise beam energies are used in output naming')
    parser.add_argument('--isolation-cone-size', type=float, default=2.5,
                       help='Isolation cone size used for model input feature (default: 2.5)')
    parser.add_argument('--isolation-cone-sizes', type=float, nargs='+', default=[1.5, 2.0, 2.5],
                       help='Isolation cone sizes to include in isolation-fraction input plots')
    parser.add_argument('--signal-generator-statuses', type=int, nargs='+', default=[1],
                       help='Generator-status values considered final-state for selecting scattered electron (default: 1)')
    args = parser.parse_args()

    file_paths = args.input_files

    p_tot, rot_y_angle, rot_x_angle = func.determine_boost(
        args.beam_electrons,
        args.beam_protons,
        args.crossing_angle,
    )

    print(f"Loading data from {len(file_paths)} files...")
    events = load_data(file_paths)

    print("Processing features...")
    X_df, y, truth_has_scattered_event, efficiency_counts = process_features(
        events,
        p_tot,
        rot_y_angle,
        rot_x_angle,
        isolation_cone_size=args.isolation_cone_size,
        isolation_cone_sizes=args.isolation_cone_sizes,
        signal_generator_statuses=args.signal_generator_statuses,
    )

    # True event-level kinematics for the full, unfiltered event sample.
    all_event_ids = np.arange(len(events), dtype=int)
    all_true_q2 = ak.to_numpy(ak.fill_none(ak.firsts(events["InclusiveKinematicsTruth.Q2"]), 0.0))
    all_true_x = ak.to_numpy(ak.fill_none(ak.firsts(events["InclusiveKinematicsTruth.x"]), 0.0))

    tag = naming_tag(args.campaign_tag, args.beam_electrons, args.beam_protons)
    model_out_path = Path(args.model_out)
    model_out_path = model_out_path.with_name(f"{model_out_path.stem}_{tag}{model_out_path.suffix}")
    model_out_path.parent.mkdir(parents=True, exist_ok=True)

    pictures_dir = Path("pictures") / f"train_{tag}"
    pictures_dir.mkdir(parents=True, exist_ok=True)

    eop_plot_out = pictures_dir / f"{model_out_path.stem}_input_E_over_p.png"
    iso_plot_out = pictures_dir / f"{model_out_path.stem}_input_isolation_fraction_scan.png"
    acop_plot_out = pictures_dir / f"{model_out_path.stem}_input_acoplanarity.png"
    grouped_plot_out = pictures_dir / f"{model_out_path.stem}_input_leadingpt_pt_charge.png"

    plot_e_over_p_distribution(X_df, y, str(eop_plot_out))
    plot_isolation_distributions(X_df, y, args.isolation_cone_sizes, str(iso_plot_out))
    plot_acoplanarity_distribution(X_df, y, str(acop_plot_out))
    plot_leading_pt_pt_charge_distributions(X_df, y, str(grouped_plot_out))

    print(f"Saved E/p input distribution to {eop_plot_out}")
    print(f"Saved isolation-fraction input distributions to {iso_plot_out}")
    print(f"Saved acoplanarity input distribution to {acop_plot_out}")
    print(f"Saved leading-pt/pt/charge input distributions to {grouped_plot_out}")

    print(f"Training model with {len(X_df)} samples...")
    (
        model,
        X_train,
        X_val,
        y_train,
        y_val,
        event_info_val,
        val_pred,
        n_train_events,
        n_val_events,
        val_events,
        val_truth_flags,
    ) = train_model(X_df, y, truth_has_scattered_event=truth_has_scattered_event)

    summary_out = model_out_path.with_name(f"{model_out_path.stem}_summary.txt")
    print_and_save_training_summary(
        n_train_events,
        n_val_events,
        len(X_train),
        len(X_val),
        int(y_train.sum()),
        int(y_val.sum()),
        str(summary_out),
    )
    print(f"Saved training summary to {summary_out}")

    efficiency_chain_out = model_out_path.with_name(f"{model_out_path.stem}_efficiency_chain.txt")
    print_and_save_efficiency_chain(
        efficiency_counts,
        n_train_events,
        n_val_events,
        int(y_train.sum()),
        int(y_val.sum()),
        str(efficiency_chain_out),
    )
    print(f"Saved efficiency chain to {efficiency_chain_out}")

    print(f"Saving model to {model_out_path}...")
    model.save_model(str(model_out_path))

    val_out = model_out_path.with_name(f"{model_out_path.stem}_val.npz")
    np.savez_compressed(
        val_out,
        X_val=X_val,
        y_val=y_val,
        val_pred=val_pred,
        event_Q2=event_info_val["Q2"].to_numpy(),
        event_x=event_info_val["x"].to_numpy(),
        event_y=event_info_val["y"].to_numpy(),
        event_id_val=event_info_val["event_id"].to_numpy(dtype=int),
        val_events=val_events,
        truth_has_scattered_event_val=(
            np.asarray(val_truth_flags, dtype=bool)
            if val_truth_flags is not None
            else np.asarray([], dtype=bool)
        ),
        all_val_events=val_events,
        all_val_true_q2=all_true_q2[val_events],
        all_val_true_x=all_true_x[val_events],
        all_val_has_scattered=(
            np.asarray(truth_has_scattered_event[val_events], dtype=bool)
            if truth_has_scattered_event is not None
            else np.asarray([], dtype=bool)
        ),
        feature_names=np.asarray(FEATURE_COLUMNS),
    )
    print(f"Saved validation arrays to {val_out}")
    print("Done!")

if __name__ == '__main__':
    main()

