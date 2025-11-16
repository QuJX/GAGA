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
from equivariant_diffusion.utils import remove_mean_with_mask, assert_correctly_masked
import torch
import pickle
from qm9.utils import compute_mean_mad
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import glob

def validate_sampling_step_size(value):
    ivalue = int(value)
    if 1000 % ivalue != 0:
        raise argparse.ArgumentTypeError(f"sampling_step_size must divide 1000 evenly. Got {ivalue}.")
    return ivalue


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def mle_std_zero_centered(x: torch.Tensor) -> torch.Tensor:
    """
    Estimate standard deviation using MLE for a zero-centered tensor,
    treating it as a flattened 1D vector.

    Args:
        x (torch.Tensor): A zero-centered tensor of any shape.

    Returns:
        torch.Tensor: A scalar tensor with the estimated standard deviation.
    """
    # Flatten the tensor and compute MLE of variance
    x_flat = x.reshape(-1)
    variance_mle = torch.mean(x_flat ** 2)
    std_mle = torch.sqrt(variance_mle)
    return std_mle

def Variance_Estimate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_qm9_max_550",
                        help='Specify model path')
    parser.add_argument('--test_name', type=str, required=True, help='QM9/GEOM')
    

    eval_args, unparsed_args = parser.parse_known_args()


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
    results = []
    output_dir = f'MLE_ZCM_{eval_args.test_name}/'

    os.makedirs(output_dir, exist_ok=True)
    grouped_results = defaultdict(list)

    for ind, data in enumerate(tqdm(train_loader, desc="Processing molecules")):
        # args, data, model_dp, device, dtype, property_norms, t_list

        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)
        one_hot = remove_mean_with_mask(one_hot, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        x, h, delta_log_px = generative_model.normalize(x, h, node_mask)

        z_0 = torch.cat((x, h['categorical']), dim = -1)
        B, N, F = z_0.shape
        
        for b in range(B):
            mask = data['atom_mask'].to(device)[b].bool()
            z = z_0[b][mask]  # [num_atoms, F]
            num_atoms = z.shape[0]

            variance_pos = mle_std_zero_centered(z[:, :3])
            variance_atom_type = mle_std_zero_centered(z[:, 3:])
            variance_all = mle_std_zero_centered(z)

            grouped_results[num_atoms].append({
                'id': ind * B + b,
                'variance_pos': float(variance_pos.detach().cpu()),
                'variance_atom_type': float(variance_atom_type.detach().cpu()),
                'variance_all': float(variance_all.detach().cpu()),
                })
        
        if ind > 3:
            break
    # Save results to separate CSVs by atom count
    # Flatten and group all rows by molecule id
    for num_atoms, rows in grouped_results.items():
        all_rows = []
        all_rows.extend(rows)

        df = pd.DataFrame(all_rows)

        # Ensure proper ordering per molecule
        df = df.sort_values(by=['id'])

        # Save full data (grouped by id) to one file
        out_path = os.path.join(output_dir, f'results_atoms_{num_atoms}.csv')
        df.to_csv(out_path, index=False)
        print(f"Saved grouped results for {df['id'].nunique()} molecules to {out_path}")
    
    # Folder containing the CSV files
    folder_path = output_dir  # Replace with your actual folder path

    # Use glob to find all matching CSV files
    csv_files = glob.glob(os.path.join(folder_path, 'results_atoms_*.csv'))

    # Collect all variance_all values
    variance_all_list = []

    for file in csv_files:
        df = pd.read_csv(file)
        variance_all_list.extend(df['variance_all'].values)

    # Convert to a pandas Series for convenient statistics
    variance_all_series = pd.Series(variance_all_list)

    # Compute aggregate statistics
    mean_variance = variance_all_series.mean()
    std_variance = variance_all_series.std()
    min_variance = variance_all_series.min()
    max_variance = variance_all_series.max()

    print(f"Mean variance_all: {mean_variance}")
    print(f"Standard deviation: {std_variance}")
    print(f"Min variance_all: {min_variance}")
    print(f"Max variance_all: {max_variance}")

if __name__ == "__main__":
    Variance_Estimate()