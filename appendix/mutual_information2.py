import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations


def calculate_mi_and_cmi(df, condition_column):
    """
    Calculate mutual information and conditional mutual information for all pairs of columns,
    conditioning on a specified column.
    """
    columns = df.columns.tolist()

    n = len(columns)
    mi_matrix = np.zeros((n, n))
    cmi_matrix = np.zeros((n, n))

    condition_data = df[condition_column].values.reshape(-1, 1)

    for i, j in combinations(range(n), 2):
        col1 = df.iloc[:, i].values.reshape(-1, 1)
        col2 = df.iloc[:, j].values.reshape(-1, 1)

        # Calculate unconditional mutual information
        mi = mutual_info_regression(col1, col2.ravel())[0]
        mi_matrix[i, j] = mi
        mi_matrix[j, i] = mi

        # Calculate conditional mutual information
        mi_1c = mutual_info_regression(np.hstack([col1, condition_data]), col2.ravel())[0]
        mi_2c = mutual_info_regression(np.hstack([col2, condition_data]), col1.ravel())[0]
        mi_c = mutual_info_regression(condition_data, col1.ravel())[0]

        cmi = 0.5 * (mi_1c + mi_2c - mi_c)
        cmi_matrix[i, j] = cmi
        cmi_matrix[j, i] = cmi

    mi_df = pd.DataFrame(mi_matrix, index=columns, columns=columns)
    cmi_df = pd.DataFrame(cmi_matrix, index=columns, columns=columns)

    return mi_df, cmi_df


# Example usage
np.random.seed(42)  # for reproducibility
df = pd.DataFrame(np.random.rand(1000, 5), columns=['A', 'B', 'C', 'D', 'E'])
condition_column = 'B'  # The column we'll condition on

# Add some correlations
df['B'] = df['A'] + np.random.normal(0, 0.1, 1000)
df['C'] = df['B'] + np.random.normal(0, 0.1, 1000)
condition_column = 'B'  # The column we'll condition on

mi_matrix, cmi_matrix = calculate_mi_and_cmi(df, condition_column)

print("Unconditional Mutual Information:")
print(mi_matrix)
print("\nConditional Mutual Information (conditioned on {}):".format(condition_column))
print(cmi_matrix)

# Calculate and print the difference
diff_matrix = mi_matrix - cmi_matrix
print("\nDifference (MI - CMI):")
print(diff_matrix)