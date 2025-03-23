import numpy as np
import pandas as pd
import argparse
import time
import re
import os
import sys
import matplotlib.pyplot as plt
import ruptures as rpt
from multiprocessing import Pool, cpu_count
from functools import partial

def plot_specific_config(config_df, target_cache, target_mem, phases):
    """
    Plot insn_rate vs insn_sum data for a specific configuration with segmentation points
    
    Parameters:
        config_df (DataFrame): DataFrame containing the performance profile data
        target_cache (int): Target cache configuration to plot
        target_mem (int): Target memory bandwidth configuration to plot
        phases: The phases for this config
    """
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot the instruction rate data
    plt.scatter(config_df['insn_sum'], config_df['insn_rate'], color='b', alpha=0.7, label='Instruction Rate')
    
    # Add vertical lines for segmentation points
    for phase_idx, phase in enumerate(phases):
        x_val = phase['start_insn']
        if phase_idx == 0:
            plt.axvline(x=x_val, color='r', linestyle='--', label='Segmentation Points')
        else:
            plt.axvline(x=x_val, color='r', linestyle='--')
    

    # Get worst-case rates for each segment and add horizontal lines
    for phase_idx, phase in enumerate(phases):
        
        # Add horizontal line for worst-case rate
        if phase_idx == 0:
            plt.hlines(y=phase['worst_case_rate'], xmin=phase['start_insn'], xmax=phase['end_insn'], 
                      colors='g', linestyles='-', label='Worst-Case Rate')
        else:
            plt.hlines(y=phase['worst_case_rate'], xmin=phase['start_insn'], xmax=phase['end_insn'], 
                      colors='g', linestyles='-')
        
        # Add phase label
        x_pos = (phase['start_insn'] + phase['end_insn']) / 2
        cv = config_df['insn_rate'].std() / config_df['insn_rate'].mean() if config_df['insn_rate'].mean() > 0 else 0
        plt.text(x_pos, phase['worst_case_rate'] * 1.1, f"Phase {phase_idx+1}\nCV={cv:.2f}", 
                horizontalalignment='center')
    
    # Add labels and title
    plt.xlabel('Cumulative Instruction Count (insn_sum)')
    plt.ylabel('Instruction Rate (insn_rate)')
    plt.title(f'Instruction Rate vs Instruction Sum for Cache={target_cache}, Mem={target_mem}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    output_file = f'cache_{target_cache}_mem_{target_mem}_changepoint.png'
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Plot saved as {output_file}")
    
    # Print summary of phases
    print("\nPhase summary:")
    for phase_idx, phase in enumerate(phases):
        print(f"Phase {phase_idx+1}:")
        print(f"  Start Insn: {phase['start_insn']}")
        print(f"  End Insn: {phase['end_insn']}")
        print(f"  Worst-Case Rate: {phase['worst_case_rate']:.2f}")
        print(f"  Duration (insn): {phase['end_insn'] - phase['start_insn']}")
        print(f"  WCET contribution: {(phase['end_insn'] - phase['start_insn']) / phase['worst_case_rate']:.2f}")


# Function to process a single configuration (for parallel execution)
def process_single_config(args, num_phases, use_worst_case=True):
    """
    Process a single cache/mem configuration for parallel execution
    
    Parameters:
        args: Tuple containing (cache, mem, group_df)
        num_phases: Number of phases to segment into
        use_worst_case: Whether to use worst-case rate or 99.99 percentile
    
    Returns:
        Tuple of (config_key, phases)
    """
    cache, mem, group_df = args
    print(f"Processing configuration: cache={cache}, mem={mem}")
    
    # Sort by instruction sum (cumulative instruction count)
    group_df = group_df.sort_values(by='insn_sum')
    
    # Convert to array for ruptures
    signal = group_df['insn_rate'].values
    
    # Use kernel change point detection with linear kernel (faster than RBF)
    start_time = time.time()  # Start timer
    algo = rpt.KernelCPD(kernel="linear", min_size=3).fit(signal)
    change_points = algo.predict(num_phases)
    end_time = time.time()  # Stop timer
    print(f"Config cache={cache}, mem={mem} - Elapsed time: {end_time - start_time:.4f} seconds")
    
    # If change_points is empty (can happen with some signals), use Pelt algorithm instead
    if not change_points or len(change_points) <= 1:
        algo = rpt.Pelt(model="l2").fit(signal)  # l2 is faster than rbf
        change_points = algo.predict(pen=10)  # Penalty parameter

    # Make sure we have at least 2 change points (start and end)
    if not change_points or len(change_points) <= 1:
        change_points = [0, len(signal)]
    elif change_points[0] != 0:
        change_points = [0] + change_points
        
    # Ensure the last index is included
    if change_points[-1] != len(signal):
        change_points.append(len(signal))
    
    # List to store phases for this configuration
    phases = []

    start_insn = 1
    
    # Create phases based on change points
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i+1]  # Use actual endpoint without subtracting 1
        
        if start_idx >= len(group_df) or end_idx > len(group_df):
            # Adjust end_idx if it exceeds the dataframe length
            if end_idx > len(group_df):
                end_idx = len(group_df)
            # Skip if still invalid
            if start_idx >= end_idx:
                continue
        
        phase_df = group_df.iloc[start_idx:end_idx]
        
        # Skip empty phases
        if len(phase_df) <= 1:
            continue
        
        # Get start and end instruction counts
        end_insn = phase_df['insn_sum'].iloc[-1]
        
        # Calculate worst-case or percentile-based instruction rate
        if use_worst_case:
            rate = phase_df['insn_rate'].min()
        else:
            rate = phase_df['insn_rate'].quantile(0.9999)  # 99.99 percentile
        
        # Create phase information
        phase = {
            'start_insn': start_insn,
            'end_insn': end_insn,
            'worst_case_rate': rate,
            'mean_rate': phase_df['insn_rate'].mean(),
            'std_rate': phase_df['insn_rate'].std(),
            'cv': phase_df['insn_rate'].std() / phase_df['insn_rate'].mean() if phase_df['insn_rate'].mean() > 0 else 0
        }

        start_insn = end_insn + 1
        
        phases.append(phase)

    if cache == 2 and mem == 2:
        # Plot the specific configuration for the worst ratio
        plot_specific_config(group_df, cache, mem, phases)
    
    # Store phases for this configuration
    config_key = f"cache_{cache}_mem_{mem}"
    return (config_key, phases)

# Parallelized version of segment_time_series
def segment_time_series_parallel(df, num_phases, use_worst_case=True, num_workers=None):
    """
    Segment time series data into phases using ruptures change point detection.
    Processes each configuration in parallel.
    
    Parameters:
        df (DataFrame): DataFrame containing the performance profile data
        num_phases (int): Number of phases to segment into
        use_worst_case (bool): Whether to use worst-case rate or 99.99 percentile
        num_workers (int): Number of parallel workers to use, defaults to CPU count - 1
    
    Returns:
        dict: Dictionary of phases for each configuration
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free for system
    
    print(f"Starting parallel processing with {num_workers} workers")
    
    # Group by cache and memory configurations
    grouped = df.groupby(['cache', 'mem'])
    
    # Prepare tasks for parallel processing
    tasks = [(cache, mem, group_df) for (cache, mem), group_df in grouped]
    
    # Create a partial function with fixed parameters
    process_func = partial(process_single_config, num_phases=num_phases, use_worst_case=use_worst_case)
    
    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_func, tasks)
    
    # Convert results list to dictionary
    all_phases = dict(results)
    
    return all_phases


# Code to plot best number of changepoints
import matplotlib.pyplot as plt
import numpy as np

# Assuming changepoints, median_ratios, and max_ratios are already populated
# from your previous code

def plot_phase_ratios(task, changepoints, ratios):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the ratios
    ax.plot(changepoints, ratios, marker='o', linestyle='-', linewidth=2, 
            color='blue')

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set labels and title
    ax.set_xlabel('Number of Changepoints', fontsize=12)
    ax.set_ylabel(f'Ratio (Phase-Based WCET / True WCET) for {task} with (cache=5, bw=5)', fontsize=12)
    ax.set_title('Impact of Phase Count on WCET Estimation Accuracy', fontsize=14)

    # Set x-axis to start at the minimum changepoint
    ax.set_xlim(min(changepoints) - 1, max(changepoints) + 1)

    # Ensure the y-axis includes 1.0 (perfect estimation)
    min_y = min(min(ratios), 1.0) * 0.95
    max_y = max(ratios) * 1.05
    ax.set_ylim(min_y, max_y)

    # Add a horizontal line at y=1 (perfect estimation)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.8, label='Perfect Estimation')

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{task}_WCET_ratios_vs_changepoints.png', dpi=300)


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

        dir = f"{task}_phases-synthetic={synthetic_profiles}-worstcase={use_worst_case}/{2 ** cache - 1}_{72 * mem}"

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
def read_df(cache_sizes, mem_bws, task, synthetic_profiles, num_per_config, directory):
    
    df_list = []

    if not os.path.exists('all_data.csv'):
        for cache_name in cache_sizes:
            for mem_name in mem_bws:

                cache = int(np.log2(int(cache_name) + 1))
                mem = int(mem_name) // 72

                for index in range(1, num_per_config + 1):
                    if synthetic_profiles:
                        filename = f"{directory}/{task}-synth_c{cache_name}_{mem_name}.txt"
                    else:
                        filename = f"{directory}/{task}_{cache_name}_{mem_name}_perf_{index}.txt"
                    
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

                                time = float(parts[0])

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



def parse_arguments():
    """
    Parse command line arguments for the task.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Phase generation script using changepoint detection.")
    
    # Required arguments
    parser.add_argument("-n", "--task_name", type=str, required=True, 
                        help="The task to perform (string)")
    parser.add_argument("-w", "--use_worst_case", type=int, required=True, 
                        help="Whether to use worst case scenario (1/0)")
    parser.add_argument("-s", "--synthetic_profile", type=int, required=True, 
                        help="Whether to use synthetic profile (1/0)")
    parser.add_argument("-i", "--input_dir", type=str, required=True, 
                        help="Raw profiles directory (string)")
    parser.add_argument("-f", "--find_num_changepoints", type=int, required=True, 
                        help="Whether to plot WCET ratios vs number of changepoints (1/0)")
    
    # Conditional arguments
    parser.add_argument("-p", "--num_per_config", type=int, 
                        help="Number of profiles per configuration (required if synthetic_profile is false)")
    parser.add_argument("-c", "--num_changepoints", type=int, 
                        help="Number of changepoints (required if optimize_num_changepoints is false)")
    
    args = parser.parse_args()
    
    # Validate conditional arguments
    if not args.synthetic_profile and args.num_per_config is None:
        parser.error("--num_per_config is required when --synthetic_profile is false")
    
    if not args.find_num_changepoints and args.num_changepoints is None:
        parser.error("--num_changepoints is required when --find_num_changepoints is false")
    
    # Validate that input directory exists
    if not os.path.isdir(args.input_dir):
        parser.error(f"Input directory '{args.input_dir}' does not exist")
        
    return args


# Main function to run the analysis
def main():

    args = parse_arguments()
    
    # Print all arguments for demonstration
    print("Parsed arguments:")
    print(f"  Task: {args.task_name}")
    print(f"  Use worst case: {args.use_worst_case}")
    print(f"  Synthetic profile: {args.synthetic_profile}")
    print(f"  Find best num changepoints: {args.find_num_changepoints}")
    print(f"  Raw profiles directory: {args.input_dir}")
    
    # Print conditional arguments
    if not args.synthetic_profile:
        print(f"  Num per config: {args.num_per_config}")
    
    if not args.find_num_changepoints:
        print(f"  Num changepoints: {args.num_changepoints}")
    
    # Set input params
    cache_sizes = [2 ** i - 1 for i in range(1, 21)]
    mem_bws = [72 * i for i in range(1, 21)]
    task = args.task_name
    synthetic_profiles = args.synthetic_profile # use the synthetic profiles?
    if synthetic_profiles:
        num_per_config = 1
    else:
        num_per_config = args.num_per_config
    use_worst_case = args.use_worst_case # use the worst case rate? If false, 99.99 percentile will be used, can change below
    input_dir = args.input_dir

    start_time_total = time.time()  # Start timer for total execution
    
    # Set number of worker processes (adjust based on your system)
    num_workers = max(1, cpu_count() - 1)  # Use all but one CPU core
    print(f"Using {num_workers} worker processes out of {cpu_count()} available cores")
    
    # Read in df
    df = read_df(cache_sizes, mem_bws, task, synthetic_profiles, num_per_config, input_dir)
    
    # Set parameters
    # Use either fixed number of phases or optimization
    num_change_points = args.num_changepoints
    plot_ratios_vs_num_changepoints = args.find_num_changepoints
    
    if plot_ratios_vs_num_changepoints:
        # Parameters for optimization
        min_phases = 5
        max_phases = 50

        changepoints = []
        ratios = []
        
        # Optimize segmentation with parallel processing
        for cur_changepoints in range(min_phases, max_phases):

            changepoints.append(cur_changepoints)

            grouped = df[(df['cache'] == 5) & (df['mem'] == 5)]

            (_, phases) = process_single_config((5, 5, grouped), cur_changepoints, use_worst_case)
            
            wcet = 0
            # Create a DataFrame with all phase data
            for phase_idx, phase in enumerate(phases):
                phase_data = {
                    'cache': 5,
                    'mem': 5,
                    'phase': phase_idx + 1,
                    'insn_start': phase['start_insn'],
                    'insn_end': phase['end_insn'],
                    'insn_rate': phase['worst_case_rate']
                }
                wcet += (phase_data['insn_end'] - phase_data['insn_start']) / phase_data['insn_rate']
                
                    
            # Get true WCET from original data
            true_wcet = df.loc[(df['cache'] == 5) & (df['mem'] == 5)]['time'].astype(float).max()


            # Calculate the ratio
            ratios.append(wcet / true_wcet)

        plot_phase_ratios(task, changepoints, ratios)
                    
    else:
        # Fixed number of phases with parallel processing
        phases = segment_time_series_parallel(df, num_change_points, use_worst_case, num_workers)
    
        # Save results in your required format
        per_phase_wcets, true_wcets = save_phases_to_output_format(phases, df, task, 
                                                                synthetic_profiles, 
                                                                use_worst_case)
        
        # Print overall statistics
        avg_ratio = sum(p/t for p, t in zip(per_phase_wcets, true_wcets)) / len(per_phase_wcets)
        print(f"Average per-phase WCET / true WCET ratio: {avg_ratio}")

        # Convert lists to numpy arrays for easier manipulation
        true_wcets = np.array(true_wcets)
        per_phase_wcets = np.array(per_phase_wcets)

        # Create a DataFrame for easier manipulation and plotting
        comparison_df = pd.DataFrame({
            'true_wcet': true_wcets,
            'per_phase_wcet': per_phase_wcets
        })

        # Calculate the ratio
        comparison_df['ratio'] = comparison_df['per_phase_wcet'] / comparison_df['true_wcet']

        # Plot 1: Scatter plot of true_wcet vs per_phase_wcet
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.scatter(true_wcets, per_phase_wcets, alpha=0.7)
        plt.plot([min(true_wcets), max(true_wcets)], [min(true_wcets), max(true_wcets)], 'r--', label='y=x')
        plt.xlabel('True WCET')
        plt.ylabel('Per-Phase WCET')
        plt.title('Per-Phase WCET vs True WCET')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot 2: Histogram of ratios
        plt.subplot(2, 2, 2)
        plt.hist(comparison_df['ratio'], bins=20, alpha=0.7)
        plt.xlabel('Ratio (Per-Phase WCET / True WCET)')
        plt.ylabel('Frequency')
        plt.title('Distribution of WCET Ratios')
        plt.grid(True, alpha=0.3)

        # Plot 3: Resource allocation heat map
        # Check if we can create the heat map

        try:
            num_cache = len(cache_sizes)
            num_mem = len(mem_bws)
            
            # Reshape the ratio data into a 2D grid if dimensions align
            if len(comparison_df) == num_cache * num_mem:
                ratio_grid = comparison_df['ratio'].values.reshape(num_cache, num_mem)
                
                plt.subplot(2, 2, 3)
                plt.imshow(ratio_grid, cmap='viridis', interpolation='nearest')
                plt.colorbar(label='Per-Phase WCET / True WCET')
                plt.xlabel('Memory Bandwidth Index')
                plt.ylabel('Cache Size Index')
                plt.title('WCET Ratio by Resource Allocation')
                
                # Add actual values as text if the grid is not too large
                if num_cache * num_mem <= 100:  # Only add text for reasonably sized grids
                    for i in range(num_cache):
                        for j in range(num_mem):
                            plt.text(j, i, f"{ratio_grid[i, j]:.2f}", 
                                    ha="center", va="center", color="w")
        except Exception as e:
            print(f"Could not create heatmap: {e}")

        # Plot 4: Line plot showing the trend
        plt.subplot(2, 2, 4)
        plt.scatter(range(len(true_wcets)), true_wcets, color='b', label='True WCET')
        plt.scatter(range(len(per_phase_wcets)), per_phase_wcets, color='g', label='Per-Phase WCET')
        plt.xlabel('Resource Allocation Index')
        plt.ylabel('WCET')
        plt.title('WCET Trends Across Resource Allocations')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('wcet_comparison_changepoint.png', dpi=300)
        
        with open(f'{task}_summary_stats.txt', 'w') as file:
            file.write("# Summary Statistics\n")
            file.write(f"Mean Ratio: {comparison_df['ratio'].mean():.4f}\n")
            file.write(f"Median Ratio: {comparison_df['ratio'].median():.4f}\n")
            file.write(f"Min Ratio: {comparison_df['ratio'].min():.4f}\n")
            file.write(f"Max Ratio: {comparison_df['ratio'].max():.4f}\n")
            file.write(f"Standard Deviation: {comparison_df['ratio'].std():.4f}\n")
        
        # Print total execution time
        end_time_total = time.time()
        print(f"Total execution time: {end_time_total - start_time_total:.2f} seconds")

if __name__ == "__main__":
   main()