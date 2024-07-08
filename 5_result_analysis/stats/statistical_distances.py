import numpy as np
import pandas as pd
from scipy.special import rel_entr
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import wasserstein_distance

def calculate_wasserstein_distance(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.nan
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Calculate Wasserstein distance for each dimension
    distances = [wasserstein_distance(x[:, i], y[:, i]) for i in range(x.shape[1])]
    
    # Return the average distance across all dimensions
    return np.mean(distances)

# Function to calculate KL Divergence with smoothing
def calculate_kl_divergence(p, q, epsilon=1e-10):
    p = p + epsilon
    q = q + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(rel_entr(p, q))

# Function to calculate Maximum Mean Discrepancy (MMD)
def calculate_mmd(x, y, sigma=1.0):
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)

    gamma = 1.0 / (2 * sigma**2)

    kxx = np.mean(np.exp(-gamma * euclidean_distances(x, x, squared=True)))
    kyy = np.mean(np.exp(-gamma * euclidean_distances(y, y, squared=True)))
    kxy = np.mean(np.exp(-gamma * euclidean_distances(x, y, squared=True)))

    return kxx + kyy - 2 * kxy

# Updated multivariate distance calculation
def calculate_multivariate_distance(synthetic_data, original_data, method):
    if len(synthetic_data) == 0 or len(original_data) == 0:
        return np.nan
    
    synthetic_data = np.asarray(synthetic_data)
    original_data = np.asarray(original_data)
    
    if method == 'kl':
        hist_synthetic, _ = np.histogramdd(synthetic_data, bins=[10]*synthetic_data.shape[1], density=True)
        hist_original, _ = np.histogramdd(original_data, bins=[10]*original_data.shape[1], density=True)
        return calculate_kl_divergence(hist_synthetic.flatten(), hist_original.flatten())
    elif method == 'wasserstein':
        return calculate_wasserstein_distance(synthetic_data, original_data)
    elif method == 'mmd':
        return calculate_mmd(synthetic_data, original_data)
    else:
        raise ValueError(f"Unknown method: {method}")

# Updated main calculation loop
def calculate_distances(scaled_df, grouping_columns, features):
    results = {}
    
    for group, group_data in scaled_df[scaled_df['source'] == 'synthetic'].groupby(grouping_columns):
        synthetic_data = group_data
        original_data = scaled_df[scaled_df['source'] == 'original']
        
        # Count samples
        n_samples = len(synthetic_data)
        
        # Multivariate calculations
        synthetic_multivariate = synthetic_data[features].values
        original_multivariate = original_data[features].values

        print(f"\nCalculating for group: {group}")
        print(f"Original data shape: {original_multivariate.shape} || Synthetic data shape: {synthetic_multivariate.shape}")
        
        group_results = {
            'n_samples': n_samples,
            'multivariate_kl': calculate_multivariate_distance(synthetic_multivariate, original_multivariate, 'kl'),
            'multivariate_wasserstein': calculate_multivariate_distance(synthetic_multivariate, original_multivariate, 'wasserstein'),
            'multivariate_mmd': calculate_multivariate_distance(synthetic_multivariate, original_multivariate, 'mmd')
        }

        print(f"Results for group {group}: {group_results}")
        
        # Handle different grouping levels
        if isinstance(group, tuple):
            current_level = results
            for level in group[:-1]:
                if level not in current_level:
                    current_level[level] = {}
                current_level = current_level[level]
            current_level[group[-1]] = group_results
        else:
            results[group] = group_results
    
    return results

# Updated function to convert nested dictionary to DataFrame
def nested_dict_to_df(d, index_names):
    rows = []
    def process_dict(current_dict, current_row):
        if isinstance(current_dict, dict):
            if all(isinstance(v, (int, float)) for v in current_dict.values()):
                # This is a leaf node with metric values
                rows.append(current_row + list(current_dict.values()))
            else:
                for key, value in current_dict.items():
                    new_row = current_row + [key]
                    process_dict(value, new_row)
        else:
            # This handles the case where a value might be a single number
            rows.append(current_row + [current_dict])
    
    process_dict(d, [])
    df = pd.DataFrame(rows, columns=index_names + ['n_samples', 'KL Divergence', 'Wasserstein', 'MMD'])
    return df
