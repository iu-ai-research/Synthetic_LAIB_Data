import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import euclidean_distances
from numba import jit
from tqdm import tqdm
import multiprocessing

def calculate_adjusted_pvalues(pvalues):
    """
    Calculate adjusted p-values using the Benjamini-Hochberg procedure.
    
    Args:
    - pvalues: List or array of p-values
    
    Returns:
    - Adjusted p-values
    """
    pvalues = np.array(pvalues)
    n = len(pvalues)
    ranks = stats.rankdata(pvalues)
    adjusted_pvalues = pvalues * n / ranks
    adjusted_pvalues = np.minimum(1, np.minimum.accumulate(adjusted_pvalues[::-1])[::-1])
    return adjusted_pvalues

def bootstrap_sample(args):
    """
    Performs a single bootstrap sample calculation.
    
    Args:
    - args: A tuple containing (combined_data, n_samples_x, n_samples_y, statistic_func, kwargs)
    
    Returns:
    - The statistic calculated on the bootstrap sample
    """
    combined, n, m, statistic_func, kwargs = args
    np.random.seed()  # Important for multiprocessing
    boot_x = combined[np.random.choice(len(combined), n, replace=True)]
    boot_y = combined[np.random.choice(len(combined), m, replace=True)]
    return statistic_func(boot_x, boot_y, **kwargs)

def calculate_bootstrap_stats(statistic_func, x, y, n_bootstrap=1000, **kwargs):
    print('Calculating Bootstrap Distributions for CI and Effect Size...')
    combined = np.vstack([x, y])
    n, m = len(x), len(y)
    
    # Use all available cores
    num_cores = multiprocessing.cpu_count() - 1
    
    # Prepare arguments for bootstrap_sample
    args = [(combined, n, m, statistic_func, kwargs)] * n_bootstrap
    
    # Parallel bootstrap using multiprocessing with progress bar
    with multiprocessing.Pool(num_cores) as pool:
        bootstrap_stats = list(tqdm(
            pool.imap(bootstrap_sample, args),
            total=n_bootstrap,
            desc="Bootstrap Progress"
        ))
    
    return np.array(bootstrap_stats)

# def bootstrap_statistic(args):
#     statistic_func, combined, n, m, kwargs = args
#     np.random.seed()  # Important for multiprocessing
#     boot_x = combined[np.random.choice(len(combined), n, replace=True)]
#     boot_y = combined[np.random.choice(len(combined), m, replace=True)]
#     return statistic_func(boot_x, boot_y, **kwargs)

# Kolmogorov-Smirnov statistic calculation
def ks_statistic(x, y):
    """
    Calculates the Kolmogorov-Smirnov statistic for two multivariate samples.
    
    Args:
    - x, y: The two datasets to compare
    
    Returns:
    - The KS statistic
    """
    n, m = len(x), len(y)
    z = np.vstack([x, y])
    idx = np.argsort(z, axis=0)
    cdf_x = np.sum(idx < n, axis=1) / n
    cdf_y = np.sum(idx >= n, axis=1) / m
    return np.max(np.abs(cdf_x - cdf_y))

# Fast Anderson-Darling statistic calculation (optimized with Numba)
@jit(nopython=True)
def fast_ad_statistic(D, n, m):
    """
    Calculates the Anderson-Darling statistic for two multivariate samples.
    This function is optimized using Numba for faster computation.
    
    Args:
    - D: Pairwise distance matrix
    - n, m: Sample sizes of the two datasets
    
    Returns:
    - The Anderson-Darling statistic
    """
    N = n + m
    ix = np.zeros(N)
    iy = np.zeros(N)
    
    for i in range(N):
        ix[i] = np.sum(D[i, :n] <= D[i, i]) - (i < n)
        iy[i] = np.sum(D[i, n:] <= D[i, i]) - (i >= n)
    
    ad = (1 / (n * m)) * np.sum((ix * (N - iy) - (n - ix) * iy)**2 / ((ix + iy) * (N - ix - iy) + 1e-8))
    return ad

@jit(nopython=True)
def calculate_distance_matrix(x, y):
    """
    Calculates the pairwise distance matrix for the combined dataset.
    
    Args:
    - x, y: The two datasets to compare
    
    Returns:
    - The pairwise distance matrix
    """
    Z = np.vstack((x, y))
    N = len(Z)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            D[i, j] = D[j, i] = np.sum((Z[i] - Z[j])**2)
    return D

# Leave function outside of multivariate_anderson_darling_test for multiprocessing
def ad_statistic(x, y):
    """
    Wrapper function to calculate the Anderson-Darling statistic.
    
    Args:
    - x, y: The two datasets to compare
    
    Returns:
    - The Anderson-Darling statistic
    """
    D = calculate_distance_matrix(x, y)
    return fast_ad_statistic(D, len(x), len(y))

# Anderson-Darling permutation function
def ad_permutation(args):
    """
    Performs a single permutation for the Anderson-Darling test.
    
    Args:
    - args: A tuple containing (combined_data, n_samples_x)
    
    Returns:
    - The Anderson-Darling statistic for this permutation
    """
    combined, n = args
    np.random.shuffle(combined)
    return ad_statistic(combined[:n], combined[n:])

# Fast KL divergence calculation (optimized with Numba)
@jit(nopython=True)
def fast_kl_divergence(x, y, k, epsilon):
    """
    Calculates the KL divergence between two multivariate samples.
    This function is optimized using Numba for faster computation.
    
    Args:
    - x, y: The two datasets to compare
    - k: Number of nearest neighbors to consider
    - epsilon: Small constant to avoid division by zero
    
    Returns:
    - The estimated KL divergence
    """
    n, m = len(x), len(y)
    d = x.shape[1]
    
    kl = 0
    for i in range(n):
        dx = np.inf
        dy = np.inf
        nx = 0
        ny = 0
        for j in range(n):
            dist = np.sum((x[i] - x[j])**2)
            if dist < dx:
                dx = dist
                nx = 1
            elif dist == dx:
                nx += 1
        for j in range(m):
            dist = np.sum((x[i] - y[j])**2)
            if dist < dy:
                dy = dist
                ny = 1
            elif dist == dy:
                ny += 1
        
        nx = max(nx, epsilon)
        ny = max(ny, epsilon)
        kl += np.log(ny) - np.log(nx)
    
    kl = kl / n + d * np.log(m / (n - 1))
    return max(kl, epsilon)

# KL divergence permutation function
def kl_permutation(args):
    """
    Performs a single permutation for the KL divergence test.
    
    Args:
    - args: A tuple containing (combined_data, n_samples_x, k, epsilon)
    
    Returns:
    - The KL divergence for this permutation
    """
    combined, n, k, epsilon = args
    combined_copy = combined.copy()  # Create a copy
    np.random.shuffle(combined_copy)
    return fast_kl_divergence(combined_copy[:n], combined_copy[n:], k, epsilon)

# Multivariate Kolmogorov-Smirnov Test
def multivariate_ks_test(x, y, min_samples=5, n_bootstrap=2, alpha=0.05): #20000
    """
    Performs a multivariate Kolmogorov-Smirnov test.
    
    Args:
    - x, y: The two datasets to compare
    - min_samples: Minimum number of samples required for the test
    - n_bootstrap: Number of bootstrap iterations
    - alpha: Significance level
    
    Returns:
    - observed: The observed KS statistic
    - p_value: The p-value of the test
    - ci_lower, ci_upper: Confidence interval bounds
    - effect_size: The effect size
    """
    print('Calculating KS Test')

    if len(x) < min_samples or len(y) < min_samples:
        print(f"Warning: Not enough samples for KS test. x: {len(x)}, y: {len(y)}")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    observed = ks_statistic(x, y)
    
    # Approximate p-value using the asymptotic distribution
    n, m = len(x), len(y)
    n_eff = n * m / (n + m)
    p_value = np.exp(-2 * n_eff * observed**2)
    
    # Calculate CI and effect size
    bootstrap_stats = calculate_bootstrap_stats(ks_statistic, x, y, n_bootstrap=n_bootstrap)
    ci_lower, ci_upper = np.percentile(bootstrap_stats, [alpha/2 * 100, (1 - alpha/2) * 100])
    effect_size = (observed - np.mean(bootstrap_stats)) / (np.std(bootstrap_stats) if np.std(bootstrap_stats) != 0 else 1e-8)
    
    return observed, p_value, ci_lower, ci_upper, effect_size

# Multivariate KL Divergence Test
def multivariate_kl_divergence_test(x, y, num_permutations=2, k=5, epsilon=1e-10, min_samples=5, n_bootstrap=2):#50 10
    """
    Performs a multivariate KL divergence test.
    
    Args:
    - x, y: The two datasets to compare
    - num_permutations: Number of permutations for the test
    - k: Number of nearest neighbors for KL divergence estimation
    - epsilon: Small constant to avoid division by zero
    - min_samples: Minimum number of samples required for the test
    
    Returns:
    - observed: The observed KL divergence
    - p_value: The p-value of the test
    - ci_lower, ci_upper: Confidence interval bounds
    - effect_size: The effect size
    """
    print('Calculating KL Test')
    
    if len(x) < min_samples or len(y) < min_samples:
        print(f"Warning: Not enough samples for KL divergence test. x: {len(x)}, y: {len(y)}")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    observed = fast_kl_divergence(x, y, k, epsilon)
    combined = np.vstack([x, y])
    n = len(x)
    
    # Use all available cores
    num_cores = multiprocessing.cpu_count() - 1
    
    # Parallel permutation test using multiprocessing with progress bar
    with multiprocessing.Pool(num_cores) as pool:
        permutation_stats = list(tqdm(
            pool.imap(kl_permutation, [(combined, n, k, epsilon)] * num_permutations),
            total=num_permutations,
            desc="KL Permutation Progress"
        ))
    
    p_value = np.mean(np.array(permutation_stats) >= observed)
    
    # Calculate CI and effect size
    bootstrap_stats = calculate_bootstrap_stats(fast_kl_divergence, x, y, n_bootstrap=n_bootstrap, k=k, epsilon=epsilon)
    ci_lower, ci_upper = np.percentile(bootstrap_stats, [2.5, 97.5])
    effect_size = (observed - np.mean(bootstrap_stats)) / (np.std(bootstrap_stats) if np.std(bootstrap_stats) != 0 else 1e-8)
    
    return observed, p_value, ci_lower, ci_upper, effect_size

# Multivariate Anderson-Darling Test
def multivariate_anderson_darling_test(x, y, num_permutations=2, min_samples=5, n_bootstrap=2, alpha=0.05):   # 40 10
    """
    Performs a multivariate Anderson-Darling test with early stopping.
    
    Args:
    - x, y: The two datasets to compare
    - num_permutations: Maximum number of permutations for the test
    - min_samples: Minimum number of samples required for the test
    - n_bootstrap: Number of bootstrap samples for CI and effect size
    - alpha: Significance level for early stopping
    
    Returns:
    - observed: The observed Anderson-Darling statistic
    - p_value: The p-value of the test
    - ci_lower, ci_upper: Confidence interval bounds
    - effect_size: The effect size
    """
    print('Calculating Anderson-Darling Test')

    if len(x) < min_samples or len(y) < min_samples:
        print(f"Warning: Not enough samples for Anderson-Darling test. x: {len(x)}, y: {len(y)}")
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    observed = ad_statistic(x, y)
    combined = np.vstack([x, y])
    n = len(x)
    
    # Use all available cores
    num_cores = 5
    
    # Early stopping parameters
    min_permutations = 100
    max_permutations = num_permutations
    
    # Parallel permutation test using multiprocessing with progress bar and early stopping
    with multiprocessing.Pool(num_cores) as pool:
        permutation_stats = []
        for i, result in enumerate(tqdm(
            pool.imap(ad_permutation, [(combined, n)] * max_permutations),
            total=max_permutations,
            desc="AD Permutation Progress"
        )):
            permutation_stats.append(result)
            
            # Check for early stopping after minimum permutations
            if i + 1 >= min_permutations:
                p_value = np.mean(np.array(permutation_stats) >= observed)
                if p_value < alpha / 10 or p_value > 1 - alpha / 10:
                    break
            
            # Stop if maximum permutations reached
            if i + 1 == max_permutations:
                break
    
    p_value = np.mean(np.array(permutation_stats) >= observed)
    
    # Calculate CI and effect size
    bootstrap_stats = calculate_bootstrap_stats(ad_statistic, x, y, n_bootstrap=n_bootstrap)
    ci_lower, ci_upper = np.percentile(bootstrap_stats, [alpha/2 * 100, (1 - alpha/2) * 100])
    effect_size = (observed - np.mean(bootstrap_stats)) / (np.std(bootstrap_stats) if np.std(bootstrap_stats) != 0 else 1e-8)
    
    return observed, p_value, ci_lower, ci_upper, effect_size

# This function is the core of the analysis. 
# It iterates through different groupings of the data, 
# applies the three statistical tests (KS, KL, and AD) to each group, 
# and organizes the results into a nested dictionary structure.
def calculate_distances(scaled_df, grouping_columns, features):
    """
    Calculates statistical distances between synthetic and original data for different groupings.
    
    Args:
    - scaled_df: DataFrame containing both synthetic and original data
    - grouping_columns: Columns to group by
    - features: Features to use for distance calculations
    
    Returns:
    - A nested dictionary containing results for each group and test
    """
    results = {}
    all_pvalues = {'ks': [], 'kl': [], 'ad': []}
    
    for group, group_data in scaled_df[scaled_df['source'] == 'synthetic'].groupby(grouping_columns):
        synthetic_data = group_data
        original_data = scaled_df[scaled_df['source'] == 'original']
        
        n_samples = len(synthetic_data)
        
        synthetic_multivariate = synthetic_data[features].values
        original_multivariate = original_data[features].values

        print(f"Calculating for group: {group}")
        print(f"Synthetic data shape: {synthetic_multivariate.shape}")
        print(f"Original data shape: {original_multivariate.shape}")
        
        # Perform tests
        ks_stat, ks_p, ks_ci_lower, ks_ci_upper, ks_effect = multivariate_ks_test(synthetic_multivariate, original_multivariate)
        kl_stat, kl_p, kl_ci_lower, kl_ci_upper, kl_effect = multivariate_kl_divergence_test(synthetic_multivariate, original_multivariate)
        ad_stat, ad_p, ad_ci_lower, ad_ci_upper, ad_effect = multivariate_anderson_darling_test(synthetic_multivariate, original_multivariate)

        # Collect p-values
        all_pvalues['ks'].append(ks_p)
        all_pvalues['kl'].append(kl_p)
        all_pvalues['ad'].append(ad_p)

        group_results = {
            'n_samples': n_samples,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'ks_ci_lower': ks_ci_lower,
            'ks_ci_upper': ks_ci_upper,
            'ks_effect_size': ks_effect,
            'kl_statistic': kl_stat,
            'kl_p_value': kl_p,
            'kl_ci_lower': kl_ci_lower,
            'kl_ci_upper': kl_ci_upper,
            'kl_effect_size': kl_effect,
            'ad_statistic': ad_stat,
            'ad_p_value': ad_p,
            'ad_ci_lower': ad_ci_lower,
            'ad_ci_upper': ad_ci_upper,
            'ad_effect_size': ad_effect,
        }


        # Ensure group_results is always a dictionary
        if not isinstance(group_results, dict):
            group_results = {'value': group_results}

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
    
    # Calculate adjusted p-values
    adjusted_pvalues = {
        'ks': calculate_adjusted_pvalues(all_pvalues['ks']),
        'kl': calculate_adjusted_pvalues(all_pvalues['kl']),
        'ad': calculate_adjusted_pvalues(all_pvalues['ad'])
    }

    # Add adjusted p-values to results
    i = 0
    for group in results:
        if isinstance(results[group], dict):
            results[group]['ks_adjusted_p_value'] = adjusted_pvalues['ks'][i]
            results[group]['kl_adjusted_p_value'] = adjusted_pvalues['kl'][i]
            results[group]['ad_adjusted_p_value'] = adjusted_pvalues['ad'][i]
        else:
            for subgroup in results[group]:
                results[group][subgroup]['ks_adjusted_p_value'] = adjusted_pvalues['ks'][i]
                results[group][subgroup]['kl_adjusted_p_value'] = adjusted_pvalues['kl'][i]
                results[group][subgroup]['ad_adjusted_p_value'] = adjusted_pvalues['ad'][i]
        i += 1

    print(results)
    return results

# This helper function converts the nested dictionary of results into a pandas DataFrame, 
# which is easier to work with for further analysis and visualization.
def nested_dict_to_df(d, index_names):
    """
    Converts a nested dictionary of results into a pandas DataFrame.
    
    Args:
    - d: Nested dictionary of results
    - index_names: Names for the index levels
    
    Returns:
    - A pandas DataFrame with multi-level index
    """
    rows = []
    all_keys = set()

    def process_dict(current_dict, current_row):
        if isinstance(current_dict, dict):
            if all(isinstance(v, (int, float, np.number)) or pd.isna(v) for v in current_dict.values()):
                all_keys.update(current_dict.keys())
                rows.append(current_row + [current_dict])
            else:
                for key, value in current_dict.items():
                    new_row = current_row + [key]
                    process_dict(value, new_row)
        else:
            # Handle non-dict values (e.g., float)
            rows.append(current_row + [{'value': current_dict}])
    
    process_dict(d, [])
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=index_names + ['data'])
    
    # Expand the 'data' column
    expanded = pd.DataFrame(df['data'].tolist(), index=df.index)
    
    # Combine the index columns with the expanded data
    result = pd.concat([df[index_names], expanded], axis=1)
    
    return result

# This function adds boolean columns to the DataFrame indicating whether each test 
# result is statistically significant based on the provided alpha level.
def add_significance(df, alpha=0.05):
    """
    Adds significance indicators to the DataFrame based on p-values and adjusted p-values.
    
    Args:
    - df: DataFrame containing test results
    - alpha: Significance level
    
    Returns:
    - DataFrame with added significance indicators
    """
    for test in ['ks', 'kl', 'ad']:
        if f'{test}_p_value' in df.columns:
            df[f'{test.upper()} Significant'] = df[f'{test}_p_value'] < alpha
        if f'{test}_adjusted_p_value' in df.columns:
            df[f'{test.upper()} Adjusted Significant'] = df[f'{test}_adjusted_p_value'] < alpha
    return df