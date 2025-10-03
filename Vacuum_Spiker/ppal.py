from subprocess import call
import datetime
import os
import pandas as pd

# Main script to launch all SNN experiments.


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

#Función que realiza la llamada:

def run_experiment(nu1, nu2, neuron, threshold, decay, expansion, resolution, path, recurrence, epochs):
    """
    Run a single experiment and log execution time and parameters.
    """
    start = datetime.datetime.now()
    call([
        'python', 'ejecute_experimentation.py',
        '-n1', str(nu1),
        '-n2', str(nu2),
        '-n', str(neuron),
        '-th', str(threshold),
        '-d', str(decay),
        '-a', str(expansion),
        '-r', str(resolution),
        '-p', str(path),
        '-e', str(epochs),
        '-rc', recurrence
    ])
    end = datetime.datetime.now()
    # Log execution time and parameters
    with open('log.txt', 'a') as log:
        nuu1 = str(nu1).replace(',', ';')
        nuu2 = str(nu2).replace(',', ';')
        epoochs = str(epochs).replace(',', ';')
        entry = f'{end},{str(end - start)},{path},{nuu1},{nuu2},{neuron},{threshold},{decay},{expansion},{resolution},{epoochs}\n'
        log.write(entry)

# Collect all input file paths (adapt to the folder where input files are placed)
paths = get_file_paths('input')

# Initialize log file if not present
with open('log.txt', 'w') as log:
    log.write('date,duration,path,nu1,nu2,neurons_b,threshold,decay,expansion,resolution,epochs\n')

# Parameter grids for experiments
nus1 = [(0.1, -0.1), (-0.1, -0.1), (-0.1, 0.1), (0.1, 0.1)]
nus2 = nus1.copy()
neurons = [100,2000]
thresholds = [-62, -55, -40]
decays = [100, 150, 200]
expansions = [1]
resolutions = [0.1, 0.001]
epochs = [1, 2, 3, 4, 5]
recurrences = ['True', 'False']


# Iterate over all combinations of parameters and run experiments
for path in paths:
    for nu1 in nus1:
        for neuron in neurons:
            for threshold in thresholds:
                for decay in decays:
                    for expansion in expansions:
                        for resolution in resolutions:
                            for recurrence in recurrences:
                                if recurrence == 'True':
                                    for nu2 in nus2:
                                        run_experiment(nu1, nu2, neuron, threshold, decay, expansion, resolution, path, recurrence, epochs)
                                elif recurrence == 'False':
                                    run_experiment(nu1, nus2[0], neuron, threshold, decay, expansion, resolution, path, recurrence, epochs)

                                    
