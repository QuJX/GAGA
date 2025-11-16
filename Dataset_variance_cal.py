import pandas as pd
import glob
import os

# Folder containing the CSV files
folder_path = 'MLE_ZCM_QM9/'  # Replace with your actual folder path

# Use glob to find all matching CSV files
csv_files = glob.glob(os.path.join(folder_path, 'results_atoms_*.csv'))

# Collect all variance_all values
variance_all_list = []
variance_pos_list = []
variance_atom_type_list = []

for file in csv_files:
    df = pd.read_csv(file)
    variance_all_list.extend(df['variance_all'].values)
    variance_pos_list.extend(df['variance_pos'].values)
    variance_atom_type_list.extend(df['variance_atom_type'].values)

# Convert to a pandas Series for convenient statistics
variance_all_series = pd.Series(variance_all_list)
variance_pos_series = pd.Series(variance_pos_list)
variance_atom_type_series = pd.Series(variance_atom_type_list)

# Compute aggregate statistics
mean_variance = variance_all_series.mean()
std_variance = variance_all_series.std()
min_variance = variance_all_series.min()
max_variance = variance_all_series.max()

print(f"Mean variance_all: {mean_variance}")
print(f"Standard deviation: {std_variance}")
print(f"Min variance_all: {min_variance}")
print(f"Max variance_all: {max_variance}")

# pos
mean_variance = variance_pos_series.mean()
std_variance = variance_pos_series.std()
min_variance = variance_pos_series.min()
max_variance = variance_pos_series.max()

print(f"Mean variance_pos: {mean_variance}")
print(f"Standard deviation: {std_variance}")
print(f"Min variance_pos: {min_variance}")
print(f"Max variance_pos: {max_variance}")

# atom_type
mean_variance = variance_atom_type_series.mean()
std_variance = variance_atom_type_series.std()
min_variance = variance_atom_type_series.min()
max_variance = variance_atom_type_series.max()

print(f"Mean variance_atom_type: {mean_variance}")
print(f"Standard deviation: {std_variance}")
print(f"Min variance_atom_type: {min_variance}")
print(f"Max variance_atom_type: {max_variance}")
