# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from configs.datasets_config import get_dataset_info
import os
from os.path import join
from qm9 import dataset
from qm9.models import get_model
import torch
import pickle
from qm9.utils import compute_mean_mad
from train_test import Get_z_t
import numpy as np
from scipy.stats import kstest
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def validate_sampling_step_size(value):
    ivalue = int(value)
    if 1000 % ivalue != 0:
        raise argparse.ArgumentTypeError(f"sampling_step_size must divide 1000 evenly. Got {ivalue}.")
    return ivalue

def mutual_info_independence_score(X, n_pairs=20, k_neighbor = 3, random_state=42):
    """
    Estimate average mutual information between random feature pairs.
    Lower score implies higher independence.
    """
    np.random.seed(random_state)
    F = X.shape[1]
    mi_vals = []
    for _ in range(n_pairs):
        i, j = np.random.choice(F, 2, replace=False)
        mi = mutual_info_regression(X[:, [i]], X[:, j], n_neighbors = k_neighbor)[0]
        mi_vals.append(mi)

    return np.mean(mi_vals)  # Lower is more independent

import numpy as np
from scipy.stats import kstest

def gaussianity_score(X, mean=None, variance=None, alpha=0.05, k_neighbor=3):
    """
    Tests Gaussianity via K-S and Mutual Information tests:
    - One-sample Kolmogorov–Smirnov on flattened standardized data vs N(0,1)
    - MI-based independence tests on both [N, F] and [F, N]

    Returns a dict of scores and pass/fail flags.
    NOTE: Because we standardize first and then test vs N(0,1),
    the KS p-values correspond to a fully-specified null (no parameter estimation).
    """
    X = np.asarray(X)
    N, F = X.shape

    # Standardize X
    if mean is None:
        mean = np.mean(X, axis=0)
    if variance is None:
        variance = np.var(X, axis=0)
    std = np.sqrt(variance)
    # Avoid division by zero if a feature has zero variance
    std = np.where(std == 0, 1.0, std)

    X_std = (X - mean) / std

    # KS test on flattened standardized data vs standard normal
    ks_stat, ks_pval = kstest(X_std.flatten(), 'norm')  # defaults to mean=0, std=1

    # MI tests on [N, F] (features) and [F, N] (samples)
    mi_feat = mutual_info_independence_score(X_std, k_neighbor=k_neighbor, n_pairs=X.shape[0] * 2)
    mi_samp = mutual_info_independence_score(X_std.T, k_neighbor=k_neighbor, n_pairs=X.shape[0] * 2)

    return {
        "KS_stat": ks_stat,
        "KS_pvalue": ks_pval,
        "MI_score_features": mi_feat,
        "MI_score_samples": mi_samp,
        "KS_passed": ks_pval > alpha,
        "Low_MI_features_passed": mi_feat < 0.1,
        "Low_MI_samples_passed": mi_samp < 0.1,
        "Overall_passed": (ks_pval > alpha) and (mi_feat < 0.1) and (mi_samp < 0.1)
    }

def noised_data_distribution_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_qm9_shortcut",
                        help='Specify model path')
    parser.add_argument('--sampling_step_size',
        type=validate_sampling_step_size,
        default=1,
        help='Step size for sampling, must divide 1000 evenly.')
    # parser.add_argument('--sampling_name', type=str, required=True, help='DDPM, DDIM, etc.')
    parser.add_argument('--test_name', type=str, required=True, help='QM9/GEOM')
    

    eval_args, unparsed_args = parser.parse_known_args()
    print(f"Sampling Step Size: {eval_args.sampling_step_size}")


    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
        
    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'): 
        args.normalization_factor = 1      # already in args, 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'    # already in args, sum

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)  # charge scale 9
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    # prop_dist is None
    # node_dist is a class, sample(): sample one N from p(N); calculate logP(N) of one N.

    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    else:
        property_norms = None
    generative_model.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy' # ema_decay 0.9999
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device) # where the model is stored
    generative_model.load_state_dict(flow_state_dict)

    train_loader = dataloaders['train']
    output_dir = f'identity_to_gaussian_{eval_args.test_name}/'

    if 'QM9' in eval_args.test_name:
        pos_variance = 1.380 # pre-calculated via first step
        atom_type_variance = 0.086
    elif 'GEOM' in eval_args.test_name:
        pos_variance = 2.393
        atom_type_variance = 0.049


    os.makedirs(output_dir, exist_ok=True)
    grouped_results = defaultdict(list)
    gaussianity_test_result = {}

    for ind, data in enumerate(tqdm(train_loader, desc="Processing molecules")):
        # args, data, model_dp, device, dtype, property_norms, t_list
        z_t_dict, atom_mask = Get_z_t(args=args, data=data, model_dp=generative_model, device=device, dtype=dtype, 
                property_norms=property_norms, t_list = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])

        for t, (z_t, alpha_t, sigma_t) in z_t_dict.items():
            if isinstance(z_t, torch.Tensor):
                z_t = z_t.squeeze(0)
            B, N, F = z_t.shape

            pos_expected_variance = alpha_t ** 2 * pos_variance + sigma_t ** 2
            atom_type_expected_variance = alpha_t ** 2 * atom_type_variance + sigma_t ** 2

            for b in range(B):
                mask = atom_mask[b].bool()
                z = z_t[b][mask].detach().cpu()  # [num_atoms, F]
                num_atoms = z.shape[0]
                pos = z[:, :3]
                atom_type = z[:, 3:]

                if num_atoms < 4:
                    continue
                gaussianity_test_result_pos = gaussianity_score(pos, mean=0, variance = pos_expected_variance, alpha=0.05, k_neighbor = 2)
                gaussianity_test_result = {}
                gaussianity_test_result['id'] = ind * B + b
                gaussianity_test_result['timestep'] = t
                gaussianity_test_result['pos_passed'] = gaussianity_test_result_pos['Overall_passed']
                if 'QM9' in eval_args.test_name:
                    gaussianity_test_result_atom_type = gaussianity_score(atom_type, mean=0, variance = atom_type_expected_variance, alpha=0.05, k_neighbor = 3)
                    gaussianity_test_result['atom_type_passed'] = gaussianity_test_result_atom_type['Overall_passed']
                elif 'GEOM' in eval_args.test_name:
                    gaussianity_test_result['atom_type_passed'] = True
                    # Because GEOM dataset has large number of classes, which makes the gaussianity test extremely complex.
                    # And after Zero-Mean, the one-hot vectors will be normalized, which will have less variance and zero mean for most dims.
                    # So here, we set the atom_type_passed as True by default, which is an emprically evaluated choice with less computational complexity. 
                grouped_results[num_atoms].append(gaussianity_test_result.copy())
    # Save results to separate CSVs by atom count
    # Flatten and group all rows by molecule id
    for num_atoms, rows in grouped_results.items():
        all_rows = []
        all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        # Ensure proper ordering per molecule
        df = df.sort_values(by=['id', 'timestep'])

        # Save full data (grouped by id) to one file
        out_path = os.path.join(output_dir, f'results_atoms_{num_atoms}.csv')
        df.to_csv(out_path, index=False)

if __name__ == "__main__":
    noised_data_distribution_test()