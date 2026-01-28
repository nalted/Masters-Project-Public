import awkward
import numpy as np


def match_particle(reco_rec, reco_sim, barrel_clu, barrel_mc, barrel_E, endcapP_clu, endcapP_mc, endcapP_E, endcapN_clu, endcapN_mc, endcapN_E, n_reco):
    barrel_E_out = [0.0] * n_reco # columns of zeroes that are as long as the number of reconstructed particles
    endcapN_E_out = [0.0] * n_reco
    endcapP_E_out = [0.0] * n_reco

    mc_to_reco = {} # empty dictionary
    for rec_index, mc_index in zip(reco_rec, reco_sim): # zip used as more efficient than for loops here
        mc_to_reco[mc_index] = rec_index # fills dictionary with mc_index as key and rec_index as value

    # barrel 
    for clu_index, mc_index in zip(barrel_clu, barrel_mc):
        if mc_index in mc_to_reco: # checks if the mc_index is in the dictionary -> means its matchable to reco particle
            reco_index = mc_to_reco[mc_index] 
            barrel_E_out[reco_index] = barrel_E[clu_index] 
    
    # negative endcap
    for clu_index, mc_index in zip(endcapN_clu, endcapN_mc):
        if mc_index in mc_to_reco:
            reco_index = mc_to_reco[mc_index]
            endcapN_E_out[reco_index] = endcapN_E[clu_index]

    # positive endcap
    for clu_index, mc_index in zip(endcapP_clu, endcapP_mc):
        if mc_index in mc_to_reco:
            reco_index = mc_to_reco[mc_index]
            endcapP_E_out[reco_index] = endcapP_E[clu_index]

def calculate_isolation(eta, phi, cluster_E, cone_size = 0.4):
    isolation_E = [] # creates empty list

    for event_eta, event_phi, event_E in zip(eta, phi, cluster_E):
        event_iso = []
        n_particles = len(event_eta)

        for i in range(n_particles):         # logic here is that you find isolation for all particles,
            d_eta = event_eta - event_eta[i] # so model can learn about non-scattered electrons that pass this metric
            d_phi = event_phi - event_phi[i]

            d_phi = ak.where(d_phi > np.pi, d_phi - (2 * np.pi), d_phi) # correction for phi values to be within -pi to pi region
            d_phi = ak.where(d_phi < -np.pi, d_phi + (2 * np.pi), d_phi)

            dR = np.sqrt(d_eta**2 + d_phi**2)

            iso_winners = (dR < cone_size) 
            iso_energy = ak.sum(event_E[iso_winners]) # sum for particles in event

            event_iso.append(iso_energy)

        isolation_E.append(event_iso)
    
    return ak.Array(isolation_E)

def find_greatest_pt(pt_array):
    for particle in range(pt_array):
        max_index = ak.argmax(pt_array[particle])
        is_leading[particle, max_index]
        return is_leading

