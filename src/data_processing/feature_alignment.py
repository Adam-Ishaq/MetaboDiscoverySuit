import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_diagnosis(file_path):
    # 1. Load the data
    df = pd.read_csv(file_path, index_col=0)
    
    # 2. Flexible Group Identification
    def identify_group(name):
        name = str(name).upper()
        if 'NAIVE' in name: return 'Naive'
        if 'SEMI' in name: return 'Semi'
        if 'QC' in name: return 'QC'
        return 'Other'

    # Apply grouping based on the sample_id (the index)
    groups = [identify_group(idx) for idx in df.index]
    df['Group_Label'] = groups
    
    print(f"--- Dataset Statistics ---")
    print(f"Samples: {df.shape[0]}")
    print(f"Features: {df.shape[1] - 1}") # Subtracting the label column
    
    # Separate features from the label for analysis
    features = df.drop(columns=['Group_Label'])
    
    # 3. Sparsity Check
    total_missing = features.isnull().sum().sum()
    sparsity_pct = (total_missing / features.size) * 100
    print(f"Total Dataset Sparsity: {sparsity_pct:.2f}%")

    if total_missing == 0:
        print("⚠️ Warning: No missing values detected. Ensure 'feature_table.csv' contains NaNs.")
        return

    # 4. MNAR Check: Correlation between Mean Intensity and Missingness
    # We use log scale because metabolomics intensities vary by orders of magnitude
    feature_means = features.mean(skipna=True)
    missing_counts = features.isnull().sum()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=np.log10(feature_means + 1), y=missing_counts, alpha=0.4)
    plt.title("MNAR Diagnosis: Intensity vs. Missingness")
    plt.xlabel("Log10(Mean Intensity)")
    plt.ylabel("Number of Missing Samples")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # 5. Group-wise Sparsity (Fixed for Deprecation)
    print("\n--- Sparsity by Group ---")
    for name, group_df in df.groupby('Group_Label'):
        group_features = group_df.drop(columns=['Group_Label'])
        group_sparsity = group_features.isnull().mean().mean() * 100
        print(f"{name}: {group_sparsity:.2f}%")

if __name__ == "__main__":
    csv_path = r'C:\Masters Thesis\Metabolomics\MyRepository\MetaboDiscoverySuit\results\feature_table.csv'
    if os.path.exists(csv_path):
        run_diagnosis(csv_path)
    else:
        print(f"File not found at: {csv_path}")