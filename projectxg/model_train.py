import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import awkward as ak
import uproot
import xgboost as xgb
import vector
import argparse
import functions as func

filter_name=[ # uses wildcards (*) to get all branches that start with the following names
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
    trees = [uproot.open(file)["events"] for file in file_paths] # loops through file paths to get the trees
    events_list = [tree.arrays(filter_name = filter_name, library = "ak") for tree in trees] # gets arrays from each tree according to the filter names
    events = ak.concatenate(events_list, axis = 0) # creates one big final array
    return events

def process_features(events):
    mc_pdg = events["MCParticles.PDG"] # pdg == 11 used to identify electrons
    mc_genstat = events["MCParticles.generatorStatus"] # partGenStat == 1 used to identify final state particles
    
    px = events["ReconstructedParticles.momentum.x"]
    py = events["ReconstructedParticles.momentum.y"]
    pz = events["ReconstructedParticles.momentum.z"]
    q = events["ReconstructedParticles.charge"]
    energy = events["ReconstructedParticles.energy"]

    xAngle = -0.025 # necessary for correcting the crossing angle
    cos_theta = np.cos(xAngle)
    sin_theta = np.sin(xAngle)

    # chosen to just correct the crossing angle, and not do the full boost, as not necessary unless i 
    # calculate quantities like Q^2, x, y, etc
    px_rot = px *cos_theta - pz * sin_theta 
    pz_rot = px * sin_theta + pz * cos_theta
    py_rot = py

    barrel_E = events["EcalBarrelClusters.energy"]
    endcapP_E = events["EcalEndcapPClusters.energy"]
    endcapN_E = events["EcalEndcapNClusters.energy"]

    vector.register_awkward() # vector used instead of ROOT.Math.PxPyPzEvector for compatibility with awkward
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

    # association indexes
    reco_to_mc_rec = events["ReconstructedParticleAssociations.recID"] 
    reco_to_mc_sim = events["ReconstructedParticleAssociations.simID"] 

    barrel_clu_index = events["EcalBarrelClusterAssociations.recID"] 
    barrel_mc_index = events["EcalBarrelClusterAssociations.simID"] 

    endcapP_clu_index = events["EcalEndcapPClusterAssociations.recID"]
    endcapP_mc_index = events["EcalEndcapPClusterAssociations.simID"]

    endcapN_clu_index = events["EcalEndcapNClusterAssociations.recID"]
    endcapN_mc_index = events["EcalEndcapNClusterAssociations.simID"]
    
    n_reco = ak.num(px)
    
    results = [ # matches calo clusters to reconstructed particles, creates array of calo energies per reco particle
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
    # assigns names to each element in the results list
    barrel_E_per_reco = ak.Array([result[0] for result in results])
    endcapP_E_per_reco = ak.Array([result[1] for result in results])
    endcapN_E_per_reco = ak.Array([result[2] for result in results])

    matched_calo_E = ak.where( # uses ak.where instead of for loops for speed
        barrel_E_per_reco > 0, # goes in order and chooses barrel first, if not barrel, then positive endcap, and then negative
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

    isolation_E = func.calculate_isolation(eta, phi, matched_calo_E, cone_size = 0.4) # uses cone size = 0.4 as its same as ElectronID example
    isolation_frac = matched_calo_E / (isolation_E + matched_calo_E)

    is_leading = func.find_greatest_pt(pt)

    labels = []
    for mc_ids, pdgs, gens in zip(reco_to_mc_sim, mc_pdg, mc_genstat):
        target_mc = -1 # default value if no electron found
        for index, (pdg, gen) in enumerate(zip(pdgs, gens)):
            if pdg == 11 and gen == 1:
                target_mc = index
                break
        label = ak.Array([1 if mc_index == target_mc else 0 for mc_index in mc_ids])
        labels.append(label)
    
    labels = ak.Array(labels) # list to single awkward array

    # flatten is necessary to convert into correct format for pandas dataframe
    E_over_p_features = ak.flatten(matched_E_over_p)
    isolation_features = ak.flatten(isolation_frac)
    is_leading_features = ak.flatten(is_leading)
    charge_features = ak.flatten(q)
    label_features = ak.flatten(labels)

    mask = ak.flatten(matched_calo_E) > 0 # mask to filter out particles with no matched calo energy
    E_over_p_features = E_over_p_features[mask]
    isolation_features = isolation_features[mask]
    is_leading_features = is_leading_features[mask]
    charge_features = charge_features[mask]
    label_features = label_features[mask]

    # creates the dataframes for features and labels, the features it learns on are listed in X_df
    X_df = pd.DataFrame({
        "E_over_p": ak.to_numpy(E_over_p_features),
        "isolation_frac": ak.to_numpy(isolation_features),
        "is_leading": ak.to_numpy(is_leading_features, dtype=int),
        "charge": ak.to_numpy(charge_features)
    })    
    Y_df = ak.to_numpy(label_features)

    return X_df, Y_df

def train_model(X, Y):
    # np.asarray to ensure all in numpy array format for xgboost
    X_array = np.asarray(X[["E_over_p", "isolation_frac", "is_leading", "charge"]])
    Y_array = np.asarray(Y)

    dtrain = xgb.DMatrix(X_array, label=Y_array)
    
    # only scale_pos_weight and objective are motivated, REMEMBER TO TWEAK LATER
    params = {
        'objective' : 'binary:logistic',
        'eval_metric' : 'auc',
        'eta' : 0.1,
        'max_depth' : 6, 
        'subsample' : 0.8,
        'colsample_bytree' : 1.0,
        'scale_pos_weight' : 4.5
    }

    watch = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=200, evals=watch)
    
    return model

# main() so that it can be ran from terminal + easy to train different models for different
# beam configs
def main():
    parser = argparse.ArgumentParser(description="Train XGBoost electron ID model")
    parser.add_argument('--model-out', type=str, required=True,
                       help='Output filename for trained model (e.g., xgb_18x275_28jan2026.json)')
    parser.add_argument('--input-files', type=str, nargs='+', required=True,
                       help='Input ROOT files for training (e.g., *.root or file1.root file2.root ...)')
    args = parser.parse_args()

    # terminal argument example : python model_train.py --model-out xgb_18x275_28jan2026.json --input-files /home/user321/rootfiles/training/18x275/*.root
    
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

