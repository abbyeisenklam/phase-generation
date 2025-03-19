import numpy as np
import pandas as pd
import time
import re
import os
import sys
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from functools import partial

def process_single_config_gmm(args, num_clusters, use_worst_case=True):
    """
    Process a single cache/mem configuration using GMM clustering
    
    Parameters:
        args: Tuple containing (cache, mem, group_df)
        num_clusters: Number of clusters for GMM
        use_worst_case: Whether to use worst-case rate or 99.99 percentile
    
    Returns:
        Tuple of (config_key, phases)
    """
    cache, mem, group_df = args
    print(f"Processing configuration: cache={cache}, mem={mem}")
    
    # Sort by instruction sum (cumulative instruction count)
    group_df = group_df.sort_values(by='insn_sum')
    
    # Make a copy of the DataFrame for clustering
    cluster_df = group_df.copy()
    
    # Select features for clustering
    features = ['L3_req', 'L3_miss', 'insn_rate', 'insn_sum']
    X = cluster_df[features].values
    
    # Standardize the features to have zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit GMM
    start_time = time.time()  # Start timer
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    
    # Handle potential empty clusters
    try:
        gmm.fit(X_scaled)
        # Predict cluster labels
        cluster_labels = gmm.predict(X_scaled)
    except Exception as e:
        print(f"Error fitting GMM for cache={cache}, mem={mem}: {e}")
        # Fall back to simple equal-sized partitioning
        cluster_labels = np.array_split(np.arange(len(cluster_df)), num_clusters)
        cluster_labels = np.concatenate([np.full(len(split), i) for i, split in enumerate(cluster_labels)])
    
    end_time = time.time()  # Stop timer
    print(f"Config cache={cache}, mem={mem} - GMM clustering elapsed time: {end_time - start_time:.4f} seconds")
    
    # Add cluster labels to the DataFrame
    cluster_df['cluster'] = cluster_labels

    # Sort by insn_sum to ensure temporal ordering
    cluster_df = cluster_df.sort_values(by='insn_sum')

    # Initialize phases
    phases = []

    # Create bucket labels based on insn_sum (every 100,000 instructions)
    bucket_size = 10000000
    cluster_df['bucket'] = (cluster_df['insn_sum'] // bucket_size) * bucket_size

    # Find the most common cluster in each bucket
    bucket_clusters = {}
    for bucket, group in cluster_df.groupby('bucket'):
        # Get the most common cluster in this bucket
        cluster_counts = group['cluster'].value_counts()
        most_common_cluster = cluster_counts.idxmax()
        bucket_clusters[bucket] = most_common_cluster

    # Assign the most common cluster of each bucket to all points in that bucket
    cluster_df['smoothed_cluster'] = cluster_df['bucket'].map(bucket_clusters)

    # Get the sequence of smoothed cluster labels
    smoothed_sequence = cluster_df['smoothed_cluster'].values

    # Find change points where the smoothed cluster changes
    change_points = [0]  # Start with the first point
    for i in range(1, len(smoothed_sequence)):
        if smoothed_sequence[i] != smoothed_sequence[i-1]:
            change_points.append(i)
    change_points.append(len(smoothed_sequence))  # Add the last point

    start_insn = 1
    
    # Create phases based on change points
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i+1]
        
        # Get the phase data
        phase_df = cluster_df.iloc[start_idx:end_idx]
        
        # Skip empty phases
        if len(phase_df) <= 1:
            continue
        
        # Get start and end instruction counts
        end_insn = phase_df['insn_sum'].iloc[-1]
        
        # Calculate worst-case or percentile-based instruction rate
        if use_worst_case:
            rate = phase_df['insn_rate'].min()
        else:
            rate = phase_df['insn_rate'].quantile(0.0001)  # 0.01 percentile (worst case for rate where lower is worse)
        
        # Create phase information
        phase = {
            'start_insn': start_insn,
            'end_insn': end_insn,
            'worst_case_rate': rate,
            'mean_rate': phase_df['insn_rate'].mean(),
            'std_rate': phase_df['insn_rate'].std(),
            'cv': phase_df['insn_rate'].std() / phase_df['insn_rate'].mean() if phase_df['insn_rate'].mean() > 0 else 0,
            'mean_L3_req': phase_df['L3_req'].mean(),
            'mean_L3_miss': phase_df['L3_miss'].mean(),
            'cluster_id': int(phase_df['cluster'].iloc[0])  # Get the GMM cluster ID for this phase
        }

        start_insn = end_insn + 1
        
        phases.append(phase)
    
    # Make sure phases don't overlap and cover the entire range
    phases.sort(key=lambda x: x['start_insn'])
    
    # Store phases for this configuration
    config_key = f"cache_{cache}_mem_{mem}"
    return (config_key, phases)

def cluster_time_series_parallel(df, num_clusters, use_worst_case=True, num_workers=None):
    """
    Cluster time series data using GMM.
    Processes each configuration in parallel.
    
    Parameters:
        df (DataFrame): DataFrame containing the performance profile data
        num_clusters (int): Number of clusters for GMM
        use_worst_case (bool): Whether to use worst-case rate or 99.99 percentile
        num_workers (int): Number of parallel workers to use, defaults to CPU count - 1
    
    Returns:
        dict: Dictionary of phases for each configuration
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free for system
    
    print(f"Starting parallel GMM clustering with {num_workers} workers")
    
    # Group by cache and memory configurations
    grouped = df.groupby(['cache', 'mem'])
    
    # Prepare tasks for parallel processing
    tasks = [(cache, mem, group_df) for (cache, mem), group_df in grouped]
    
    # Create a partial function with fixed parameters
    process_func = partial(process_single_config_gmm, num_clusters=num_clusters, use_worst_case=use_worst_case)
    
    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_func, tasks)
    
    # Convert results list to dictionary
    all_phases = dict(results)
    
    return all_phases

def optimize_clustering_parallel(df, min_clusters=3, max_clusters=15, use_worst_case=True, num_workers=None):
    """
    Find optimal number of clusters that minimizes variation within phases.
    Processes each configuration in parallel.
    
    Parameters:
        df (DataFrame): DataFrame containing the performance profile data
        min_clusters (int): Minimum number of clusters to try
        max_clusters (int): Maximum number of clusters to try
        use_worst_case (bool): Whether to use worst-case rate or 99.99 percentile
        num_workers (int): Number of parallel workers to use, defaults to CPU count - 1
        
    Returns:
        dict: Dictionary of optimized phases for each configuration
    """
    def optimize_single_config(args):
        """Helper function to optimize a single configuration for parallel execution"""
        cache, mem, group_df = args
        print(f"Optimizing clustering for configuration: cache={cache}, mem={mem}")
        
        # Sort by instruction sum
        group_df = group_df.sort_values(by='insn_sum')
        
        # Select features for clustering
        features = ['L3_req', 'L3_miss', 'insn_rate', 'insn_sum']
        X = group_df[features].values
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        best_bic = np.inf
        best_n_clusters = min_clusters
        best_labels = None
        
        # Try different numbers of clusters
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                # Use GMM clustering
                print(f"Config cache={cache}, mem={mem} - Trying with {n_clusters} clusters")
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                gmm.fit(X_scaled)
                
                # Use BIC as the criterion for optimal clusters
                bic = gmm.bic(X_scaled)
                
                # Update best if this clustering has lower BIC
                if bic < best_bic:
                    best_bic = bic
                    best_n_clusters = n_clusters
                    best_labels = gmm.predict(X_scaled)
            except Exception as e:
                print(f"Error with n_clusters={n_clusters}: {e}")
                continue
        
        print(f"Best number of clusters for configuration cache={cache}, mem={mem}: {best_n_clusters} (BIC={best_bic:.4f})")
        
        # Use best labels or fallback if optimization failed
        if best_labels is None:
            # Fall back to simple equal-sized partitioning
            best_labels = np.array_split(np.arange(len(group_df)), min_clusters)
            best_labels = np.concatenate([np.full(len(split), i) for i, split in enumerate(best_labels)])
        
        # Add cluster labels to the DataFrame
        group_df['cluster'] = best_labels
        
        # Create phases with consecutive instruction counts
        # Sort by insn_sum to ensure temporal ordering
        group_df = group_df.sort_values(by='insn_sum')
        
        # Find change points where the cluster changes
        change_points = [0]  # Start with the first point
        cluster_sequence = group_df['cluster'].values
        for i in range(1, len(cluster_sequence)):
            if cluster_sequence[i] != cluster_sequence[i-1]:
                change_points.append(i)
        change_points.append(len(cluster_sequence))  # Add the last point
        
        phases = []
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i+1]
            
            # Get the phase data
            phase_df = group_df.iloc[start_idx:end_idx]
            
            # Skip empty phases
            if len(phase_df) <= 1:
                continue
            
            start_insn = phase_df['insn_sum'].iloc[0]
            end_insn = phase_df['insn_sum'].iloc[-1]
            
            if use_worst_case:
                rate = phase_df['insn_rate'].min()
            else:
                rate = phase_df['insn_rate'].quantile(0.0001)
            
            phase = {
                'start_insn': start_insn,
                'end_insn': end_insn,
                'worst_case_rate': rate,
                'mean_rate': phase_df['insn_rate'].mean(),
                'std_rate': phase_df['insn_rate'].std(),
                'cv': phase_df['insn_rate'].std() / phase_df['insn_rate'].mean() if phase_df['insn_rate'].mean() > 0 else 0,
                'mean_L3_req': phase_df['L3_req'].mean(),
                'mean_L3_miss': phase_df['L3_miss'].mean(),
                'cluster_id': int(phase_df['cluster'].iloc[0])
            }
            
            phases.append(phase)
        
        config_key = f"cache_{cache}_mem_{mem}"
        return (config_key, phases)
    
    # Set up parallel processing
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print(f"Starting parallel optimization with {num_workers} workers")
    
    # Group by cache and memory configurations
    grouped = df.groupby(['cache', 'mem'])
    
    # Prepare tasks for parallel processing
    tasks = [(cache, mem, group_df) for (cache, mem), group_df in grouped]
    
    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(optimize_single_config, tasks)
    
    # Convert results list to dictionary
    best_phases = dict(results)
    
    return best_phases

def plot_specific_config_gmm(df, target_cache=5, target_mem=5, num_clusters=15):
    """
    Plot insn_rate vs insn_sum data for a specific configuration with GMM clustering
    
    Parameters:
        df (DataFrame): DataFrame containing the performance profile data
        target_cache (int): Target cache configuration to plot
        target_mem (int): Target memory bandwidth configuration to plot
        num_clusters (int): Number of clusters for GMM
    """
    # Filter data for the specific configuration
    config_df = df[(df['cache'] == target_cache) & (df['mem'] == target_mem)]
    
    if len(config_df) == 0:
        print(f"No data found for configuration cache={target_cache}, mem={target_mem}")
        return
    
    print(f"Found {len(config_df)} data points for configuration cache={target_cache}, mem={target_mem}")
    
    # Sort by instruction sum (cumulative instruction count)
    config_df = config_df.sort_values(by='insn_sum')
    
    # Select features for clustering
    features = ['L3_req', 'L3_miss', 'insn_rate', 'insn_sum']
    X = config_df[features].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    
    try:
        gmm.fit(X_scaled)
        # Predict cluster labels
        cluster_labels = gmm.predict(X_scaled)
    except Exception as e:
        print(f"Error fitting GMM: {e}")
        # Fall back to simple equal-sized partitioning
        cluster_labels = np.array_split(np.arange(len(config_df)), num_clusters)
        cluster_labels = np.concatenate([np.full(len(split), i) for i, split in enumerate(cluster_labels)])
    
    # Add cluster labels to the DataFrame
    config_df['cluster'] = cluster_labels
    
    # Create buckets based on instruction count (every 1,000,000 instructions)
    bucket_size = 10000000
    config_df['bucket'] = (config_df['insn_sum'] // bucket_size) * bucket_size
    
    # Find the most common cluster in each bucket
    bucket_clusters = {}
    for bucket, group in config_df.groupby('bucket'):
        # Get the most common cluster in this bucket
        cluster_counts = group['cluster'].value_counts()
        most_common_cluster = cluster_counts.idxmax()
        bucket_clusters[bucket] = most_common_cluster
    
    # Assign the most common cluster of each bucket to all points in that bucket
    config_df['smoothed_cluster'] = config_df['bucket'].map(bucket_clusters)
    
    # Create a new figure for the plot
    plt.figure(figsize=(14, 8))
    
    # Scatter plot colored by smoothed cluster (prominent)
    plt.scatter(config_df['insn_sum'], config_df['insn_rate'], 
                         c=config_df['cluster'], cmap='plasma', 
                         alpha=0.7, s=30, label='Clusters')
    
    # Sort by insn_sum to ensure temporal ordering
    config_df = config_df.sort_values(by='insn_sum')
    
    # Get the sequence of smoothed cluster labels
    smoothed_sequence = config_df['smoothed_cluster'].values
    
    # Find change points where the smoothed cluster changes
    change_points = [0]  # Start with the first point
    for i in range(1, len(smoothed_sequence)):
        if smoothed_sequence[i] != smoothed_sequence[i-1]:
            change_points.append(i)
    change_points.append(len(smoothed_sequence))  # Add the last point
    
    # Draw vertical lines at change points and calculate worst-case rates
    phases = []
    start_insn = 1
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i+1]
        
        # Get the phase data
        phase_df = config_df.iloc[start_idx:end_idx]
        
        # Skip empty phases
        if len(phase_df) <= 1:
            continue
        
        # Get start and end instruction counts
        end_insn = phase_df['insn_sum'].iloc[-1]
        
        # Draw vertical line
        if i == 0:
            plt.axvline(x=start_insn, color='r', linestyle='--', label='Phase Boundaries')
        else:
            plt.axvline(x=start_insn, color='r', linestyle='--')
        
        # Calculate worst-case rate
        worst_case_rate = phase_df['insn_rate'].min()
        
        # Add horizontal line for worst-case rate
        if i == 0:
            plt.hlines(y=worst_case_rate, xmin=start_insn, xmax=end_insn,
                      colors='g', linestyles='-', label='Worst-Case Rate')
        else:
            plt.hlines(y=worst_case_rate, xmin=start_insn, xmax=end_insn,
                      colors='g', linestyles='-')
        
        # Add phase label
        x_pos = (start_insn + end_insn) / 2
        cv = phase_df['insn_rate'].std() / phase_df['insn_rate'].mean() if phase_df['insn_rate'].mean() > 0 else 0
        plt.text(x_pos, worst_case_rate * 1.1, 
                f"Phase {i+1}\nBucket Cluster {int(phase_df['smoothed_cluster'].iloc[0])}\nCV={cv:.2f}",
                horizontalalignment='center')
        
        # Store phase information
        phase = {
            'phase_num': i+1,
            'start_insn': start_insn,
            'end_insn': end_insn,
            'worst_case_rate': worst_case_rate,
            'mean_rate': phase_df['insn_rate'].mean(),
            'std_rate': phase_df['insn_rate'].std(),
            'cv': cv,
            'original_cluster_id': int(phase_df['cluster'].iloc[0]),
            'smoothed_cluster_id': int(phase_df['smoothed_cluster'].iloc[0]),
            'mean_L3_req': phase_df['L3_req'].mean(),
            'mean_L3_miss': phase_df['L3_miss'].mean(),
            'insn_range': end_insn - start_insn,
            'bucket_count': len(phase_df['bucket'].unique())
        }

        start_insn = end_insn + 1

        phases.append(phase)
    
    # Add the last vertical line
    plt.axvline(x=config_df['insn_sum'].iloc[-1], color='r', linestyle='--')
    
    # Add labels and title
    plt.xlabel('Cumulative Instruction Count (insn_sum)')
    plt.ylabel('Instruction Rate (insn_rate)')
    plt.title(f'GMM Clustering with Bucketing for Cache={target_cache}, Mem={target_mem}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    output_file = f'cache_{target_cache}_mem_{target_mem}_cluster.png'
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Plot saved as {output_file}")
    
    # Print summary of phases
    print("\nPhase summary:")
    for phase in phases:
        print(f"Phase {phase['phase_num']} (Smoothed Cluster {phase['smoothed_cluster_id']}):")
        print(f"  Start Insn: {phase['start_insn']}")
        print(f"  End Insn: {phase['end_insn']}")
        print(f"  Insn Range: {phase['insn_range']}")
        print(f"  Number of Buckets: {phase['bucket_count']}")
        print(f"  Worst-Case Rate: {phase['worst_case_rate']:.2f}")
        print(f"  Mean Rate: {phase['mean_rate']:.2f}")
        print(f"  Mean L3 Req: {phase['mean_L3_req']:.2f}")
        print(f"  Mean L3 Miss: {phase['mean_L3_miss']:.2f}")
        print(f"  CV: {phase['cv']:.2f}")
        print(f"  WCET contribution: {(phase['end_insn'] - phase['start_insn']) / phase['worst_case_rate']:.2f}")

def save_phases_to_output_format(phases, df, task, synthetic_profiles, use_worst_case):
    """
    Save phases according to the required output format
    
    Parameters:
        phases (dict): Dictionary of phases for each configuration
        df (DataFrame): Original DataFrame with all data
        task (str): Task name
        synthetic_profiles (bool): Whether synthetic profiles were used
        use_worst_case (bool): Whether worst case rates were used
    """
    # Prepare to collect WCETs
    per_phase_wcets = []
    true_wcets = []
    
    # Create a DataFrame with all phase data
    for config_key, config_phases in phases.items():
        cache = int(config_key.split('_')[1])
        mem = int(config_key.split('_')[3])

        dir = f"{task}_phases_synthetic={synthetic_profiles}_worstcase_rates={use_worst_case}_gmm_clustering=True/{2 ** cache - 1}_{72 * mem}"

        # Make the directory if it doesn't exist
        os.makedirs(dir, exist_ok=True)

        output_path = os.path.join(dir, 'phases.txt')

        # Calculate WCET
        wcet = 0
        for phase_idx, phase in enumerate(config_phases):
            phase_data = {
                'cache': cache,
                'mem': mem,
                'phase': phase_idx + 1,
                'insn_start': phase['start_insn'],
                'insn_end': phase['end_insn'],
                'insn_rate': phase['worst_case_rate']
            }
            wcet += (phase_data['insn_end'] - phase_data['insn_start']) / phase_data['insn_rate']
            
        per_phase_wcets.append(wcet)
            
        # Get true WCET from original data
        true_wcet = df.loc[(df['cache'] == cache) & (df['mem'] == mem)]['time'].astype(float).max()
        true_wcets.append(true_wcet)
            
        print(f"C: {cache}, BW: {mem}, per-phase wcet / true wcet: {wcet / true_wcet}")

        max_insn = df.loc[(df['cache'] == cache) & (df['mem'] == mem)]['insn_sum'].max()
        
        output_path2 = os.path.join(dir, 'wcet.txt')

        with open(output_path2, 'w') as file:
            file.write(f"{wcet}\n")
            file.write(f"{max_insn}\n")

        # Create a DataFrame for this phase and save to CSV
        phase_df = pd.DataFrame([{
            'phase': i+1,
            'insn_start': phase['start_insn'],
            'insn_end': phase['end_insn'],
            'insn_rate': phase['worst_case_rate']
        } for i, phase in enumerate(config_phases)])
        
        phase_df.to_csv(output_path, index=False, header=False, sep=',')
    
    return per_phase_wcets, true_wcets

# Load data
def read_df(cache_sizes, mem_bws, task, synthetic_profiles, num_per_config):
    
    df_list = []

    if not os.path.exists('all_data.csv'):
        for cache_name in cache_sizes:
            for mem_name in mem_bws:

                cache = int(np.log2(int(cache_name) + 1))
                mem = int(mem_name) // 72

                for index in range(1, num_per_config + 1):
                    if synthetic_profiles:
                        filename = f"{task}-synth_c{cache_name}_{mem_name}.txt"
                    else:
                        filename = f"/root/RTAS25-profile-training/{task}_profile/{task}_{cache_name}_{mem_name}_perf_{index}.txt"
                    
                    print(f"Loading file: {filename}")


                    if synthetic_profiles:
                        tmp_df = pd.read_csv(
                            filename, 
                            header=None,               # No headers in the CSV
                            usecols=[0, 4, 5, 6],      # Select columns 1, 5, 6, and 7 (0-based)
                            names=["time", "insn", "L3_req", "L3_miss"]  # Custom column names
                        )

                        tmp_df["cache"] = cache
                        tmp_df["mem"] = mem

                        tmp_df["insn_sum"] = tmp_df["insn"].cumsum()

                        tmp_df["insn_rate"] = tmp_df["insn"] / 0.01

                        tmp_df = tmp_df.loc[tmp_df["insn"] != 0]

                        df_list.append(tmp_df)

                    else:
                        with open(filename, 'r') as f:
                            lines = f.readlines()

                            data_rows = []

                            for line in lines:
                                if line.startswith("#") or not line.strip():
                                    continue

                                parts = line.strip().split()  # Simpler split
                                if len(parts) < 3:
                                    continue

                                float(parts[0])

                                count = int(parts[1].replace(",", "")) if '<not' not in parts[1] else 0
                                event_type = parts[2]

                                if "instructions" in event_type:
                                    cur_insn = count
                                    row = {'time': time, 'insn': cur_insn, 
                                        'L3_req': 0, 'L3_miss': 0, 'cache': cache, 'mem': mem,
                                    }
                                    data_rows.append(row)

                                elif "LLC-loads" in event_type and "misses" not in event_type:
                                    data_rows[-1]['L3_req'] = count

                                elif "LLC-stores" in event_type:
                                    data_rows[-1]['L3_req'] += count

                                elif "LLC-loads-misses" in event_type:
                                    data_rows[-1]['L3_miss'] = count

                            # Remove last entry since rate is skewed by program finishing in < 10 ms
                            if data_rows:
                                data_rows.pop()
                            
                            tmp_df = pd.DataFrame(data_rows)

                            # Calculate instruction rate
                            prev_time = 0.0  # Use 0 for the first row as specified
                            insn_rates = []
                            
                            for i, row in tmp_df.iterrows():
                                current_time = row['time']
                                time_diff = current_time - prev_time
                                
                                # Handle the case where time difference is 0 to avoid division by zero
                                if time_diff == 0:
                                    rate = 0
                                else:
                                    rate = row['insn'] / time_diff
                                    
                                insn_rates.append(rate)
                                prev_time = current_time
                                
                            tmp_df['insn_rate'] = insn_rates

                            tmp_df["insn_sum"] = tmp_df["insn"].cumsum()

                            tmp_df = tmp_df.loc[tmp_df["insn"] != 0]

                            df_list.append(tmp_df)
                            
                    
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv('all_data.csv')

    else:
        df = pd.read_csv("all_data.csv")
    
    return df


# Main function to run the analysis
def main():
    start_time_total = time.time()  # Start timer for total execution
    
    # Input params
    cache_sizes = [2 ** i - 1 for i in range(1, 21)]
    mem_bws = [72 * i for i in range(1, 21)]
    task = "canneal"
    synthetic_profiles = False # use the synthetic profiles?
    if synthetic_profiles:
        num_per_config = 1
    else:
        num_per_config = 100
    use_worst_case = True # use the worst case rate? If false, 99.99 percentile will be used, can change below

    # Set number of worker processes (adjust based on your system)
    num_workers = max(1, cpu_count() - 1)  # Use all but one CPU core
    print(f"Using {num_workers} worker processes out of {cpu_count()} available cores")
    
    # Read in df
    df = read_df(cache_sizes, mem_bws, task, synthetic_profiles, num_per_config)
    
    # Set parameters
    # Use either fixed number of clusters or optimization
    num_clusters = 15
    use_optimization = False
    
    # if use_optimization:
    #     # Parameters for optimization
    #     min_clusters = 3
    #     max_clusters = 20
        
    #     # Optimize clustering with parallel processing
    #     phases = optimize_clustering_parallel(df, min_clusters, max_clusters, use_worst_case, num_workers)
    # else:
    #     # Fixed number of clusters with parallel processing
    #     phases = cluster_time_series_parallel(df, num_clusters, use_worst_case, num_workers)
    
    # # Save results in your required format
    # per_phase_wcets, true_wcets = save_phases_to_output_format(phases, df, task, 
    #                                                            synthetic_profiles, 
    #                                                            use_worst_case)
    
    # # Print overall statistics
    # avg_ratio = sum(p/t for p, t in zip(per_phase_wcets, true_wcets)) / len(per_phase_wcets)
    # print(f"Average per-phase WCET / true WCET ratio: {avg_ratio}")

    # # Convert lists to numpy arrays for easier manipulation
    # true_wcets = np.array(true_wcets)
    # per_phase_wcets = np.array(per_phase_wcets)

    # # Create a DataFrame for easier manipulation and plotting
    # comparison_df = pd.DataFrame({
    #     'true_wcet': true_wcets,
    #     'per_phase_wcet': per_phase_wcets
    # })

    # # Calculate the ratio
    # comparison_df['ratio'] = comparison_df['per_phase_wcet'] / comparison_df['true_wcet']

    # # Plot 1: Scatter plot of true_wcet vs per_phase_wcet
    # plt.figure(figsize=(12, 10))

    # plt.subplot(2, 2, 1)
    # plt.scatter(true_wcets, per_phase_wcets, alpha=0.7)
    # plt.plot([min(true_wcets), max(true_wcets)], [min(true_wcets), max(true_wcets)], 'r--', label='y=x')
    # plt.xlabel('True WCET')
    # plt.ylabel('Per-Phase WCET')
    # plt.title('Per-Phase WCET vs True WCET')
    # plt.grid(True, alpha=0.3)
    # plt.legend()

    # # Plot 2: Histogram of ratios
    # plt.subplot(2, 2, 2)
    # plt.hist(comparison_df['ratio'], bins=20, alpha=0.7)
    # plt.xlabel('Ratio (Per-Phase WCET / True WCET)')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of WCET Ratios')
    # plt.grid(True, alpha=0.3)

    # # Plot 3: Resource allocation heat map
    # # Check if we can create the heat map
    # try:
    #     num_cache = len(cache_sizes)
    #     num_mem = len(mem_bws)
        
    #     # Reshape the ratio data into a 2D grid if dimensions align
    #     if len(comparison_df) == num_cache * num_mem:
    #         ratio_grid = comparison_df['ratio'].values.reshape(num_cache, num_mem)
            
    #         plt.subplot(2, 2, 3)
    #         plt.imshow(ratio_grid, cmap='viridis', interpolation='nearest')
    #         plt.colorbar(label='Per-Phase WCET / True WCET')
    #         plt.xlabel('Memory Bandwidth Index')
    #         plt.ylabel('Cache Size Index')
    #         plt.title('WCET Ratio by Resource Allocation')
            
    #         # Add actual values as text if the grid is not too large
    #         if num_cache * num_mem <= 100:  # Only add text for reasonably sized grids
    #             for i in range(num_cache):
    #                 for j in range(num_mem):
    #                     plt.text(j, i, f"{ratio_grid[i, j]:.2f}", 
    #                             ha="center", va="center", color="w")
    # except Exception as e:
    #     print(f"Could not create heatmap: {e}")

    # # Plot 4: Line plot showing the trend
    # plt.subplot(2, 2, 4)
    # plt.scatter(range(len(true_wcets)), true_wcets, color='b', label='True WCET')
    # plt.scatter(range(len(per_phase_wcets)), per_phase_wcets, color='g', label='Per-Phase WCET')
    # plt.xlabel('Resource Allocation Index')
    # plt.ylabel('WCET')
    # plt.title('WCET Trends Across Resource Allocations')
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig('wcet_comparison_cluster.png', dpi=300)
    
    # # Summary statistics
    # print("Summary Statistics:")
    # print(f"Mean Ratio: {comparison_df['ratio'].mean():.4f}")
    # print(f"Median Ratio: {comparison_df['ratio'].median():.4f}")
    # print(f"Min Ratio: {comparison_df['ratio'].min():.4f}")
    # print(f"Max Ratio: {comparison_df['ratio'].max():.4f}")
    # print(f"Standard Deviation: {comparison_df['ratio'].std():.4f}")
    
    # # Print total execution time
    # end_time_total = time.time()
    # print(f"Total execution time: {end_time_total - start_time_total:.2f} seconds")

    # Plot a specific configuration for detailed analysis
    plot_specific_config_gmm(df, 4, 1, num_clusters)

if __name__ == "__main__":
   main()
