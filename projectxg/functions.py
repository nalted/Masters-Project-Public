import awkward as ak
import numpy as np
import vector


def determine_boost(beam_E_electron, beam_E_proton, crossing_angle=-0.025):
    beam_electron = vector.obj(px=0.0, py=0.0, pz=-beam_E_electron, E=beam_E_electron)
    beam_proton = vector.obj(
        px=beam_E_proton * np.sin(crossing_angle),
        py=0.0,
        pz=beam_E_proton * np.cos(crossing_angle),
        E=beam_E_proton,
    )

    p_tot = beam_electron + beam_proton
    p_p_cm = beam_proton.boostCM_of_p4(p_tot)

    rot_y_angle = -1.0 * np.arctan2(p_p_cm.px, p_p_cm.pz)
    p_p_cm_rot_y = p_p_cm.rotateY(rot_y_angle)

    rot_x_angle = np.arctan2(p_p_cm_rot_y.py, p_p_cm_rot_y.pz)

    return p_tot, rot_y_angle, rot_x_angle


def apply_boost(px, py, pz, E, p_tot, rot_y_angle, rot_x_angle):
    particle = vector.Array(
        ak.zip(
            {
                "px": px,
                "py": py,
                "pz": pz,
                "E": E,
            }
        )
    )
    boosted = particle.boostCM_of_p4(p_tot).rotateY(rot_y_angle).rotateX(rot_x_angle)
    return boosted.px, boosted.py, boosted.pz, boosted.E


def match_via_mc_vectorised(
    reco_rec,
    reco_sim,
    barrel_clu,
    barrel_mc,
    barrel_val,
    endcapP_clu,
    endcapP_mc,
    endcapP_val,
    endcapN_clu,
    endcapN_mc,
    endcapN_val,
    n_reco,
):
    barrel_out = np.zeros(n_reco, dtype=np.float64)
    endcapN_out = np.zeros(n_reco, dtype=np.float64)
    endcapP_out = np.zeros(n_reco, dtype=np.float64)

    reco_rec_np = np.asarray(reco_rec, dtype=np.intp)
    reco_sim_np = np.asarray(reco_sim, dtype=np.intp)

    if len(reco_sim_np) == 0:
        return barrel_out, endcapN_out, endcapP_out

    max_mc = (
        max(
            reco_sim_np.max(),
            int(ak.max(barrel_mc)) if len(barrel_mc) > 0 else 0,
            int(ak.max(endcapP_mc)) if len(endcapP_mc) > 0 else 0,
            int(ak.max(endcapN_mc)) if len(endcapN_mc) > 0 else 0,
        )
        + 1
    )

    mc_to_reco = np.full(max_mc, -1, dtype=np.intp)
    mc_to_reco[reco_sim_np] = reco_rec_np

    for clu_idx_arr, mc_idx_arr, val_arr, out_arr in [
        (barrel_clu, barrel_mc, barrel_val, barrel_out),
        (endcapN_clu, endcapN_mc, endcapN_val, endcapN_out),
        (endcapP_clu, endcapP_mc, endcapP_val, endcapP_out),
    ]:
        if len(mc_idx_arr) == 0:
            continue

        mc_np = np.asarray(mc_idx_arr, dtype=np.intp)
        clu_np = np.asarray(clu_idx_arr, dtype=np.intp)
        reco_targets = mc_to_reco[mc_np]
        valid = reco_targets >= 0
        out_arr[reco_targets[valid]] = np.asarray(val_arr)[clu_np[valid]]

    return barrel_out, endcapN_out, endcapP_out


def xyz_to_eta_phi(x, y, z):
    r_t = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(r_t, z)
    eta = -np.log(np.tan(theta / 2.0))
    return eta, phi


def calculate_isolation_vectorised(
    cand_eta,
    cand_phi,
    field_eta,
    field_phi,
    field_weight,
    cone_size=2.5,
):
    c_eta = cand_eta[:, :, np.newaxis]
    c_phi = cand_phi[:, :, np.newaxis]

    f_eta = field_eta[:, np.newaxis, :]
    f_phi = field_phi[:, np.newaxis, :]

    d_eta = f_eta - c_eta
    d_phi = f_phi - c_phi
    d_phi = ak.where(d_phi > np.pi, d_phi - 2 * np.pi, d_phi)
    d_phi = ak.where(d_phi < -np.pi, d_phi + 2 * np.pi, d_phi)
    dR2 = d_eta**2 + d_phi**2

    in_cone = dR2 < cone_size**2

    weights = field_weight[:, np.newaxis, :]
    return ak.sum(ak.where(in_cone, weights, 0.0), axis=-1)


def find_greatest_pt(pt_array):
    is_leading = []
    for event_pt in pt_array:
        if len(event_pt) == 0:
            is_leading.append([])
        else:
            max_index = ak.argmax(event_pt)
            event_flags = [i == max_index for i in range(len(event_pt))]
            is_leading.append(event_flags)
    return ak.Array(is_leading)


def calc_acoplanarity(cand_px, cand_py, all_px, all_py):
    acoplanarity = []

    for ev_cand_px, ev_cand_py, ev_all_px, ev_all_py in zip(cand_px, cand_py, all_px, all_py):
        ev_cand_px = ak.to_numpy(ev_cand_px)
        ev_cand_py = ak.to_numpy(ev_cand_py)
        ev_all_px = ak.to_numpy(ev_all_px)
        ev_all_py = ak.to_numpy(ev_all_py)

        n_cand = len(ev_cand_px)
        total_px = np.sum(ev_all_px)
        total_py = np.sum(ev_all_py)
        event_acoplanarity = []

        for i in range(n_cand):
            cpx = ev_cand_px[i]
            cpy = ev_cand_py[i]
            recoil_px = total_px - cpx
            recoil_py = total_py - cpy
            cand_mag = np.sqrt(cpx**2 + cpy**2)
            recoil_mag = np.sqrt(recoil_px**2 + recoil_py**2)

            if cand_mag == 0 or recoil_mag == 0:
                event_acoplanarity.append(0.0)
                continue

            cos_dphi = (cpx * recoil_px + cpy * recoil_py) / (cand_mag * recoil_mag)
            cos_dphi = np.clip(cos_dphi, -1.0, 1.0)
            dphi = np.arccos(cos_dphi)
            event_acoplanarity.append(np.pi - dphi)

        acoplanarity.append(event_acoplanarity)

    return ak.Array(acoplanarity)