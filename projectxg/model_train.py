import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import awkward as ak
import uproot
import xgboost as xgb
import vector
import matplotlib.pyplot as plt
import argparse
import functions as func
from sklearn.model_selection import train_test_split

filter_name=[
            "ReconstructedParticles/ReconstructedParticles.*",
            "MCParticles/MCParticles.*",
            "EcalBarrelClusters/EcalBarrelClusters.*",
            "EcalEndcapPClusters/EcalEndcapPClusters.*",
            "EcalEndcapNClusters/EcalEndcapNClusters.*",
            "EcalBarrelClusterAssociations/EcalBarrelClusterAssociations.*",
            "EcalEndcapPClusterAssociations/EcalEndcapPClusterAssociations.*",
            "EcalEndcapNClusterAssociations/EcalEndcapNClusterAssociations.*",
            "ReconstructedParticleAssociations/ReconstructedParticleAssociations.*"
            ]

def load_data(file_paths):
    trees = [uproot.open(file)["events"] for file in file_paths]
    events_list = [tree.arrays(filter_name = filter_name, library = "ak") for tree in trees]
    events = ak.concatenate(events_list, axis = 0)
    return events

def process_features(events):
    mc_pdg = events["MCParticles.PDG"]
    mc_genstat = events["MCParticles.generatorStatus"]
    
    px = events["ReconstructedParticles.momentum.x"]
    py = events["ReconstructedParticles.momentum.y"]
    pz = events["ReconstructedParticles.momentum.z"]
    q = events["ReconstructedParticles.charge"]
    energy = events["ReconstructedParticles.energy"]

    xAngle = -0.025
    cos_theta = np.cos(xAngle)
    sin_theta = np.sin(xAngle)

    px_rot = px *cos_theta - pz * sin_theta
    pz_rot = px * sin_theta + pz * cos_theta
    py_rot = py

    barrel_E = events["EcalBarrelClusters.energy"]
    endcapP_E = events["EcalEndcapPClusters.energy"]
    endcapN_E = events["EcalEndcapNClusters.energy"]

    vector.register_awkward()
    reco_vector = vector.Array(ak.zip({
        "px": px_rot,
        "py": py_rot,
        "pz": pz_rot,
        "energy": energy
    }))

    pt = reco_vector.pt
    eta = reco_vector.eta
    phi = reco_vector.phi
    p = reco_vector.mag

    reco_to_mc_rec = events["ReconstructedParticleAssociations.recID"]
    reco_to_mc_sim = events["ReconstructedParticleAssociations.simID"]

    barrel_clu_index = events["EcalBarrelClusterAssociations.recID"]
    barrel_mc_index = events["EcalBarrelClusterAssociations.simID"]

    endcapP_clu_index = events["EcalEndcapPClusterAssociations.recID"]
    endcapP_mc_index = events["EcalEndcapPClusterAssociations.simID"]

    endcapN_clu_index = events["EcalEndcapNClusterAssociations.recID"]
    endcapN_mc_index = events["EcalEndcapNClusterAssociations.simID"]
    
    n_reco = ak.num(px)

    results = [
        func.match_particle(r_rec, r_sim, bcl, bmc, bE, pcl, pmc, pE, ncl, nmc, nE, nreco)
        for r_rec, r_sim, bcl, bmc, bE, pcl, pmc, pE, ncl, nmc, nE, nreco in zip(
            reco_to_mc_rec,
            reco_to_mc_sim,
            barrel_clu_index,
            barrel_mc_index,
            barrel_E,
            endcapP_clu_index,
            endcapP_mc_index,
            endcapP_E,
            endcapN_clu_index,
            endcapN_mc_index,
            endcapN_E,
            n_reco
        )
    ]
    barrel_E_per_reco = ak.Array([result[0] for result in results])
    endcapP_E_per_reco = ak.Array([result[1] for result in results])
    endcapN_E_per_reco = ak.Array([result[2] for result in results])

    barrel_E_over_p = barrel_E_per_reco / p
    endcapP_E_over_p = endcapP_E_per_reco / p
    endcapN_E_over_p = endcapN_E_per_reco / p

    matched_calo_E = ak.where(
        barrel_E_per_reco > 0,
        barrel_E_per_reco,
        ak.where(
            endcapP_E_per_reco > 0,
            endcapP_E_per_reco,
            ak.where(
                endcapN_E_per_reco > 0,
                endcapN_E_per_reco,
                0.0
        )
    ))
    matched_E_over_p = matched_calo_E / p

    isolation_E = func.calculate_isolation(eta, phi, matched_calo_E, cone_size = 0.4)
    isolation_frac = matched_calo_E / (isolation_E + matched_calo_E)

    is_leading = func.find_greatest_pt(pt)

    labels = []
    for mc_ids, pdgs, gens in zip(reco_to_mc_sim, mc_pdg, mc_genstat):
        target_mc = -1
        for index, (pdg, gen) in enumerate(zip(pdgs, gens)):
            if pdg == 11 and gen == 1:
                target_mc = index
                break
        label = ak.Array([1 if mc_index == target_mc else 0 for mc_index in mc_ids])
        labels.append(label)
    
    labels = ak.Array(labels)

    E_over_p_features = ak.flatten(matched_E_over_p)
    isolation_features = ak.flatten(isolation_frac)
    is_leading_features = ak.flatten(is_leading)
    charge_features = ak.flatten(q)
    label_features = ak.flatten(labels)

    mask = ak.flatten(matched_calo_E) > 0
    E_over_p_features = E_over_p_features[mask]
    isolation_features = isolation_features[mask]
    is_leading_features = is_leading_features[mask]
    charge_features = charge_features[mask]
    label_features = label_features[mask]

    X_df = pd.DataFrame({
        "E_over_p": ak.to_numpy(E_over_p_features),
        "isolation_frac": ak.to_numpy(isolation_features),
        "is_leading": ak.to_numpy(is_leading_features, dtype=int),
        "charge": ak.to_numpy(charge_features)
    })    
    Y_df = ak.to_numpy(label_features)

    return X_df, Y_df

def train_model(X, Y):
    X_array = np.asarray(X[["E_over_p", "isolation_frac", "is_leading", "charge"]])
    Y_array = np.asarray(Y)

    X_train, X_val, Y_train, Y_val = train_test_split(X_array, Y_array, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dval = xgb.DMatrix(X_val, label=Y_val)

    params = {
        'objective' : 'binary:logistic',
        'eval_metric' : 'auc',
        'eta' : 0.1,
        'max_depth' : 4, 
        'subsample' : 0.8,
        'colsample_bytree' : 1.0,
        'scale_pos_weight' : 4.5
    }

    watch = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=200, evals=watch, early_stopping_rounds=20)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost electron ID model")
    parser.add_argument('--model-out', type=str, required=True,
                       help='Output filename for trained model (e.g., xgb_18x275_28jan2026.json)')
    parser.add_argument('--input-files', type=str, nargs='+', required=True,
                       help='Input ROOT files for training (e.g., *.root or file1.root file2.root ...)')
    args = parser.parse_args()
    
    file_paths = args.input_files
    
    print(f"Loading data from {len(file_paths)} files...")
    events = load_data(file_paths)
    
    print("Processing features...")
    X_df, Y_df = process_features(events)
    
    print(f"Training model with {len(X_df)} samples...")
    model = train_model(X_df, Y_df)
    
    print(f"Saving model to {args.model_out}...")
    model.save_model(args.model_out)
    print("Done!")

if __name__ == '__main__':
    main()

