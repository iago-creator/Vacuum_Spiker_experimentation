from subprocess import call
import datetime
import os

# Main script to launch all experiments.

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

def run_experiment(path, batch_size, learning_rate, epoch, model, lon=None, hidden=None, latent=None, n_layer=None):
    """
    Run a single experiment with the given parameters and log the execution time.
    """
    start_time = datetime.datetime.now()
    if model == 'LSTMAutoencoder':
        call([ #This instruction is written to be run on Linux. To run it on Windows, generate a string instead of a list.
            'python', 'execute_experimentation.py',
            '-p', str(path),
            '-bs', str(batch_size),
            '-lr', str(learning_rate),
            '-e', str(epoch),
            '-m', str(model),
            '-l', str(lon),
            '-hi', str(hidden),
            '-lat', str(latent),
            '-n', str(n_layer)
        ])
    elif model == 'YildirimOzal':
        # Sequence length is required for this model
        call([
            'python', 'execute_experimentation.py',
            '-p', str(path),
            '-bs', str(batch_size),
            '-lr', str(learning_rate),
            '-e', str(epoch),
            '-m', str(model),
            '-l', str(lon)
        ])
        
    
    
    elif model == 'Conv1dAutoencoder':
        # Sequence length and n_layer are required
        call([
            'python', 'execute_experimentation.py',
            '-p', str(path),
            '-bs', str(batch_size),
            '-lr', str(learning_rate),
            '-e', str(epoch),
            '-m', str(model),
            '-l', str(lon),
            '-n', str(n_layer)
        ])
    else:
        call([
            'python', 'execute_experimentation.py'
            '-p', str(path),
            '-bs', str(batch_size),
            '-lr', str(learning_rate),
            '-e', str(epoch),
            '-m', str(model)
        ])
    
    end_time = datetime.datetime.now()
    # Log execution time and parameters
    with open('log.txt', 'a') as log:
        log_entry = f'{end_time},{str(end_time - start_time)},{path},{batch_size},{learning_rate},{epoch},{model},{lon}\n'
        log.write(log_entry)

# Initialize log file with header
with open('log.txt', 'w') as log:
    log.write('date,duration,path,batch_size,learning_rate,epoch,model,length\n')

# Collect all input file paths (adapt to the folder where input files are placed)
paths = get_file_paths('input')

# Experiment parameter grids
batch_sizes = [32, 64, 128]

learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
epochs = [10, 50, 100]
lons = [50, 100, 150, 200]
hiddens = [32, 64]
latents = [20, 50]
n_layers = [1, 2, 3]
models = ['AdaptiveZhengZhenyu','AdaptiveOhShuLih','AdaptiveCaiWenjuan','Conv1dAutoencoder','LSTMAutoencoder']
for model in models:
    for path in paths:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for epoch in epochs:
                    for n_layer in n_layers:
                        for lon in lons:
                            if model == 'Conv1dAutoencoder':
                                run_experiment(path, batch_size, learning_rate, epoch, model, lon, n_layer=n_layer)
                            elif model == 'LSTMAutoencoder':
                                for hidden in hiddens:
                                    for latent in latents:
                                        run_experiment(path, batch_size, learning_rate, epoch, model, lon, hidden, latent, n_layer)

