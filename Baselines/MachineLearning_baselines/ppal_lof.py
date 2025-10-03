# Import required libraries
from subprocess import call
import datetime
import os

# Main script to launch full kNN experimentation


def get_file_paths(base_dir):
    """
    Get all file paths recursively from a base directory.
    """
    file_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    return file_paths


def run_experiment(path, seq_length, neighbours):
    """
    Run a single kNN experiment and log execution time.
    """
    start_time = datetime.datetime.now()
    call(['python', 'exp_lof.py', '-p', str(path), '-l', str(seq_length), '-k', str(neighbours)])
    end_time = datetime.datetime.now()
    # Log execution time and parameters
    with open('log_lof.txt', 'a') as log:
        log_entry = f'{end_time},{str(end_time - start_time)},{path},{seq_length},{neighbours}\n'
        log.write(log_entry)


# Initialize log file
with open('log_lof.txt', 'w') as log:
    log.write('date,duration,path,length,neighbours\n')

# Collect all input file paths (adapt to the folder where input files are placed)
paths = get_file_paths('input')

# Experiment parameters
lons = [50, 100, 150, 200]
neighbours_list = [30, 50]

print(paths)

# Run experiments for all combinations
for path in paths:
    print(path)
    for seq_length in lons:
        print(f'Sequence length: {seq_length}')
        for neighbours in neighbours_list:
            print(f'Neighbors: {neighbours}')
            run_experiment(path, seq_length, neighbours)

