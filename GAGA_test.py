# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_model
from equivariant_diffusion.utils import assert_correctly_masked
from qm9.sampling import shortcut_sample
import torch
import time
import pickle
from qm9.utils import compute_mean_mad
from qm9.analyze import analyze_stability_for_molecules
from tqdm import tqdm

def validate_sampling_step_size(value):
    ivalue = int(value)
    if 1000 % ivalue != 0:
        raise argparse.ArgumentTypeError(f"sampling_step_size must divide 1000 evenly. Got {ivalue}.")
    return ivalue


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def GA_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_geom_drugs",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')
                        
    parser.add_argument('--sampling_step_size',
        type=validate_sampling_step_size,
        default=1,
        help='Step size for sampling, must divide 1000 evenly.')
    parser.add_argument('--shortcut', type=str, default='Gaussian')
    parser.add_argument('--test_name', type=str, required=True, help='Custom name tag for saving XYZ files.')
    parser.add_argument('--dataset_var', type=float, required=True, help='Dataset var for estimate the gaussian for approximation')
    parser.add_argument('--GA_timestep', type=int, required=True, help='which timestep for gaussian for approximation')
    parser.add_argument('--iteration', type=int, required=True, help='how many iterations are needed')
    

    eval_args, unparsed_args = parser.parse_known_args()
    print(f"Sampling Step Size: {eval_args.sampling_step_size}")


    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    
    print(args)

    # embed()
        
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
    args.batch_size = eval_args.batch_size_gen
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

    start_time = time.time()
    batch_size = eval_args.batch_size_gen
    n_samples = eval_args.n_samples

    for iter in range(eval_args.iteration):
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        for index, ind in enumerate(tqdm(range(n_samples // batch_size), desc=f"Processing {eval_args.shortcut} shortcut at {eval_args.GA_timestep} timestep")):
            # args, data, model_dp, device, dtype, property_norms, t_list
            nodesxsample = nodes_dist.sample(batch_size)
            one_hot, charges, x, node_mask = shortcut_sample(
                args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample = nodesxsample, start_point=eval_args.GA_timestep, 
                sampling_step_size=eval_args.sampling_step_size, sampling_name=eval_args.sampling_name, shortcut = eval_args.shortcut, dataset_var=eval_args.dataset_var)

            molecules['one_hot'].append(one_hot.detach().cpu())  # n, 29,5
            molecules['x'].append(x.detach().cpu())    # n, 29, 3
            molecules['node_mask'].append(node_mask.detach().cpu()) # n,29,1

            current_num_samples = (ind+1) * batch_size
            secs_per_sample = (time.time() - start_time) / current_num_samples
            print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}  # each key now is a length of 1 list, convert to length of n list
        stability_dict, rdkit_metrics = analyze_stability_for_molecules(molecules, dataset_info)
        if rdkit_metrics is not None:
            rdkit_metrics = rdkit_metrics[0]
            print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
        else:
            print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")
        print(stability_dict)
        with open(join(eval_args.model_path, f'EDM_GA-T={eval_args.GA_timestep}-Shortcut_log.txt'), 'a') as f:
            print(f"{eval_args.shortcut}-Shortcut {eval_args.GA_timestep}-0: {eval_args.test_name}; Sample size: {eval_args.n_samples}, "
                    f"Sampling Step Size: {eval_args.sampling_step_size}, "
                    f"Shortcut Method: GA, Shortcut Step: {eval_args.GA_timestep}, "
                    f"Validity: {rdkit_metrics[0]:.4f}, Uniqueness: {rdkit_metrics[1]:.4f}, Novelty: {rdkit_metrics[2]:.4f} "
                    f"{stability_dict}",
                    file=f)
        
    return stability_dict, rdkit_metrics




if __name__ == "__main__":
    GA_test()
