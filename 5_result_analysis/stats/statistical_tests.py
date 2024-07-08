import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import gamma
from tqdm import tqdm

def nearest_neighbor_distances(data, k):
    tree = cKDTree(data)
    distances, _ = tree.query(data, k=k+1)
    return distances[:, -1]

def kl_divergence_knn(original_data, synthetic_data, k=5):
    n = len(original_data)
    m = len(synthetic_data)
    d = original_data.shape[1]

    r_k = nearest_neighbor_distances(original_data, k)
    s_k = nearest_neighbor_distances(synthetic_data, k)

    # Ensure all distances are positive
    r_k = np.maximum(r_k, 1e-10)
    s_k = np.maximum(s_k, 1e-10)

    # Convert distances to densities
    const = (gamma(d / 2 + 1) / np.pi**(d / 2)) * (k / (n - 1))
    p_k = const / r_k**d
    q_k = const / s_k**d

    term1 = np.sum(np.log(p_k / q_k))
    term2 = n * np.log(m / (n - 1))

    kl_div = (term1 + term2) / n

    return max(kl_div, 0)  # Ensure non-negativity

def kl_divergence_permutation_test_knn(original_data, synthetic_data, k=5, permutations=99):
    # Adjust sample size to match the smaller dataset if synthetic data has more samples than original
    if len(original_data) < len(synthetic_data):
        synthetic_data = synthetic_data[:len(original_data)]
    elif len(synthetic_data) < len(original_data):
        original_data = original_data[:len(synthetic_data)]

    observed_kl_div = kl_divergence_knn(original_data, synthetic_data, k)
    print(f"Observed KL Divergence: {observed_kl_div}")
    
    combined_data = np.vstack([original_data, synthetic_data])
    permuted_kl_divs = []
    
    for i in tqdm(range(permutations), desc="KL Divergence Permutations"):
        np.random.shuffle(combined_data)
        permuted_original = combined_data[:len(original_data)]
        permuted_synthetic = combined_data[len(original_data):]
        permuted_kl_div = kl_divergence_knn(permuted_original, permuted_synthetic, k)
        permuted_kl_divs.append(permuted_kl_div)
    
    permuted_kl_divs = np.array(permuted_kl_divs)
    p_value = np.mean(permuted_kl_divs >= observed_kl_div)
    
    effect_size = observed_kl_div / np.sum(permuted_kl_divs) if np.sum(permuted_kl_divs) != 0 else 0
    
    return observed_kl_div, p_value, effect_size

def ecdf(data, points):
    """Compute the empirical cumulative distribution function (ECDF) at given points."""
    return np.array([np.mean(np.all(data <= point, axis=1)) for point in points])

def multivariate_ks_test(X, Y, permutations=99):
    n = len(X)
    m = len(Y)

    # Combine the data
    combined_data = np.vstack([X, Y])

    # Compute the observed test statistic
    ecdf_X = ecdf(X, combined_data)
    ecdf_Y = ecdf(Y, combined_data)
    D_nm = np.max(np.abs(ecdf_X - ecdf_Y))

    # Permutation test
    permuted_D = []
    for i in tqdm(range(permutations), desc="KS Test Permutations"):
        np.random.shuffle(combined_data)
        perm_X = combined_data[:n]
        perm_Y = combined_data[n:]
        perm_ecdf_X = ecdf(perm_X, combined_data)
        perm_ecdf_Y = ecdf(perm_Y, combined_data)
        perm_D_nm = np.max(np.abs(perm_ecdf_X - perm_ecdf_Y))
        permuted_D.append(perm_D_nm)

    # Calculate p-value
    permuted_D = np.array(permuted_D)
    p_value = np.mean(permuted_D >= D_nm)

    # Calculate effect size
    effect_size = D_nm / np.sqrt(n * m / (n + m))

    return D_nm, p_value, effect_size