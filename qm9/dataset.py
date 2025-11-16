from torch.utils.data import DataLoader
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
import os

def design_statistical_test_plan(dataloader, confidence_level=0.95, epsilon=0.05, total_test_fraction=0.2):
    """
    Given a dataloader and desired confidence level for Gaussianity testing, compute the optimal number of tests (m)
    and batch size (L) using the Central Limit Theorem and a fixed testing budget.

    Parameters:
    - dataloader: PyTorch-like dataloader providing access to the dataset.
    - confidence_level (float): Desired statistical confidence level (e.g., 0.95).
    - epsilon (float): Allowed error in estimating the Gaussian acceptance rate.
    - total_test_fraction (float): Fraction of total dataset allowed to be used for testing.

    Returns:
    - m (int): Number of test repetitions.
    - L (int): Batch size per test.
    """
    import math
    from scipy.stats import norm

    # Step 1: Compute dataset size
    dataset_size = len(dataloader.dataset)

    # Step 2: Get z-score for desired confidence
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)

    # Step 3: Compute m using conservative bound (p=0.5 worst-case)
    m = math.ceil((z_score ** 2) / (4 * epsilon ** 2))

    # Step 4: Constrain total usage to a fraction of dataset
    max_samples = int(total_test_fraction * dataset_size)
    if m > max_samples:
        m = max_samples

    # Step 5: Compute L from available data
    L = max(1, dataset_size // m)

    return m, L

def retrieve_dataloaders(cfg):

    if 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}
        
        return dataloaders, charge_scale
        

    elif 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        data_file = './data/geom/geom_drugs_30.npy'
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):

    for key in datasets:

        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None

    return datasets
