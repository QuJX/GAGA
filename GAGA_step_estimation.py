import os
import pandas as pd
from glob import glob
import numpy as np
import argparse


def average_earliest_valid_timestep_QM9(folder_path):
    """
    Given a folder of CSVs matching pattern 'results_atoms_*.csv',
    find the average of the earliest timestep for each molecule (id)
    where both pos_passed and atom_type_passed are True for two
    consecutive timesteps.
    """
    earliest_timesteps = []

    for filepath in glob(os.path.join(folder_path, 'results_atoms_*.csv')):
        df = pd.read_csv(filepath)
        df.sort_values(by=['id', 'timestep'], inplace=True)

        for mol_id, group in df.groupby('id'):
            group = group.reset_index(drop=True)
            for i in range(len(group) - 1):
                row1 = group.loc[i]
                row2 = group.loc[i + 1]
                if (row1['pos_passed'] and row1['atom_type_passed'] and
                    row2['pos_passed'] and row2['atom_type_passed']):
                    earliest_timesteps.append(row1['timestep'])
                    break  # only take the first occurrence per molecule

    if earliest_timesteps:
        mean = np.mean(earliest_timesteps)
        var = np.var(earliest_timesteps)
        return mean, var
    else:
        return None  # No molecules passed the test
    
def average_earliest_valid_timestep_GEOM(folder_path):
    """
    Given a folder of CSVs matching pattern 'results_atoms_*.csv',
    find the average of the earliest timestep for each molecule (id)
    where both pos_passed and atom_type_passed are True for two
    consecutive timesteps.
    """
    earliest_timesteps = []

    for filepath in glob(os.path.join(folder_path, 'results_atoms_*.csv')):
        df = pd.read_csv(filepath)
        df.sort_values(by=['id', 'timestep'], inplace=True)

        for mol_id, group in df.groupby('id'):
            group = group.reset_index(drop=True)
            for i in range(len(group) - 1):
                row1 = group.loc[i]
                row2 = group.loc[i + 1]
                if (row1['pos_passed'] and
                    row2['pos_passed'] ):
                    earliest_timesteps.append(row1['timestep'])
                    break  # only take the first occurrence per molecule

    if earliest_timesteps:
        mean = np.mean(earliest_timesteps)
        var = np.var(earliest_timesteps)
        return mean, var
    else:
        return None  # No molecules passed the test

def earliest_threshold_timestep_QM9(folder_path, quantile=0.95):
    """
    Returns the smallest timestep t* such that at least `quantile` proportion
    of molecules pass the consecutive tests at or before t*.
    """
    earliest_timesteps = []

    for filepath in glob(os.path.join(folder_path, 'results_atoms_*.csv')):
        df = pd.read_csv(filepath)
        df.sort_values(by=['id', 'timestep'], inplace=True)

        for mol_id, group in df.groupby('id'):
            group = group.reset_index(drop=True)
            for i in range(len(group) - 1):
                row1 = group.loc[i]
                row2 = group.loc[i + 1]
                if (row1['pos_passed'] and row1['atom_type_passed'] and
                    row2['pos_passed'] and row2['atom_type_passed']):
                    earliest_timesteps.append(row1['timestep'])
                    break

    if not earliest_timesteps:
        return None

    # Use numpy's quantile to find the minimal timestep where `quantile`% molecules have passed
    threshold = int(np.quantile(earliest_timesteps, quantile, method="nearest"))
    return threshold

def earliest_threshold_timestep_GEOM(folder_path, quantile=0.95):
    """
    Returns the smallest timestep t* such that at least `quantile` proportion
    of molecules pass the consecutive tests at or before t*.
    """
    earliest_timesteps = []

    for filepath in glob(os.path.join(folder_path, 'results_atoms_*.csv')):
        df = pd.read_csv(filepath)
        df.sort_values(by=['id', 'timestep'], inplace=True)

        for mol_id, group in df.groupby('id'):
            group = group.reset_index(drop=True)
            for i in range(len(group) - 1):
                row1 = group.loc[i]
                row2 = group.loc[i + 1]
                if (row1['pos_passed'] and
                    row2['pos_passed']):
                    earliest_timesteps.append(row1['timestep'])
                    break

    if not earliest_timesteps:
        return None

    # Use numpy's quantile to find the minimal timestep where `quantile`% molecules have passed
    threshold = int(np.quantile(earliest_timesteps, quantile, method="nearest"))
    return threshold


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='T* Estimation')
    parser.add_argument('--data_root', type=str, default='identity_to_gaussian_QM9/')
    args = parser.parse_args()

    avg_time, var = average_earliest_valid_timestep_QM9(args.data_root)
    print(f"Average earliest valid timestep: {avg_time} + {var}")
