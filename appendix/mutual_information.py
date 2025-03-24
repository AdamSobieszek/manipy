import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt


def mutual_info_continuous(X, Y, n_bins=20):
    """Calculate mutual information for continuous variables"""
    XY = np.c_[X, Y]
    c_xy = np.histogram2d(XY[:, 0], XY[:, 1], bins=n_bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)


def conditional_mutual_info_continuous(X, Y, Z, n_bins=20):
    """Calculate conditional mutual information I(X;Y|Z) for continuous variables"""
    X, Y, Z = map(np.asarray, (X, Y, Z))
    XYZ = np.c_[X, Y, Z]
    XZ = np.c_[X, Z]
    YZ = np.c_[Y, Z]

    c_xyz = np.histogramdd(XYZ, bins=n_bins)[0]
    c_xz = np.histogramdd(XZ, bins=n_bins)[0]
    c_yz = np.histogramdd(YZ, bins=n_bins)[0]
    c_z = np.histogram(Z, bins=n_bins)[0]

    I_XYZ = mutual_info_score(None, None, contingency=c_xyz.reshape(n_bins, -1))
    I_XZ = mutual_info_score(None, None, contingency=c_xz)
    I_YZ = mutual_info_score(None, None, contingency=c_yz)
    I_Z = mutual_info_score(None, None, contingency=c_z.reshape(-1, 1))

    return I_XYZ - I_XZ - I_YZ + I_Z


def conditional_mutual_info_matrix(df, condition_column, n_bins=20):
    """
    Calculate and display a matrix of conditional mutual information for all pairs of columns,
    conditioned on a selected column.

    Parameters:
    df: pandas DataFrame
    condition_column: str, name of the column to condition on
    n_bins: int, number of bins for discretization

    Returns:
    cmi_matrix: pandas DataFrame, matrix of conditional mutual information
    """
    columns = df.columns
    n = len(columns)
    cmi_matrix = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i != j and col1 != condition_column and col2 != condition_column:
                cmi = conditional_mutual_info_continuous(
                    df[col1].values, df[col2].values, df[condition_column].values, n_bins)
                cmi_matrix.loc[col1, col2] = cmi
                cmi_matrix.loc[col2, col1] = cmi
            elif i == j:
                cmi_matrix.loc[col1, col2] = mutual_info_continuous(df[col1].values, df[col1].values, n_bins)

    # Set diagonal to NaN to exclude self-information
    np.fill_diagonal(cmi_matrix.values, np.nan)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cmi_matrix, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Conditional Mutual Information'})
    plt.title(f'Conditional Mutual Information Matrix (conditioned on {condition_column})')
    plt.tight_layout()
    plt.show()

    return cmi_matrix


# Example usage
# np.random.seed(42)  # for reproducibility
df = pd.DataFrame(np.random.rand(1000, 5), columns=['A', 'B', 'C', 'D', 'E'])
# Add some correlations
df['B'] = df['A'] + np.random.normal(0, 0.1, 1000)
df['C'] = df['B'] + np.random.normal(0, 0.1, 1000)
condition_column = 'B'  # The column we'll condition on

cmi_matrix = conditional_mutual_info_matrix(df, condition_column)
print("\nConditional Mutual Information Matrix:")
print(cmi_matrix)