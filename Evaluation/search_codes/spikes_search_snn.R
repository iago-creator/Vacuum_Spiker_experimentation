###############################################################################
# Main script to generate SNN energy consumption tables
# - Iterates over all SNN hyperparameter combinations
# - Collects spike counts using count_spikes_snn function
# - Calculates MAC operations based on spike activity
# - Writes energy consumption data to CSV
###############################################################################

# Discover datasets inside the configured dataset_name path
datasets <- list.files(dataset_name)

datasets4table<-c()
nu1s4table<-c()
nu2s4table<-c()
ns4table<-c()
thresholds4table<-c()
decays4table<-c()
amplitudes4table<-c()
epochss4table<-c()
resolutions4table<-c()
recurrences4table<-c()

spike_counts <- c()

# Iterate through all hyperparameter combinations
for (dataset in datasets) {
  print(paste("Processing dataset:", dataset))
  
  for (nu1 in nu1_values) {
    for (n_neuron in neuron_counts) {
      for (threshold in threshold_values) {
        for (decay in decay_values) {
          for (amplitude in amplitude_values) {
            for (epochs in epoch_values_snn) {
              for (resolution in resolution_values) {
                for (recurrence in recurrent_flags) {
                  
                  if (recurrence == "True") {
                    # For recurrent networks, iterate over nu2 values
                    for (nu2 in nu2_values) {
                      
                      # Collect hyperparameters for this configuration
                      datasets4table<-c(datasets4table,dataset)
                      nu1s4table<-c(nu1s4table,nu1)
                      nu2s4table<-c(nu2s4table,nu2)
                      ns4table<-c(ns4table,n_neuron)
                      thresholds4table<-c(thresholds4table,threshold)
                      decays4table<-c(decays4table,decay)
                      amplitudes4table<-c(amplitudes4table,amplitude)
                      epochss4table<-c(epochss4table,epochs)
                      resolutions4table<-c(resolutions4table,resolution)
                      recurrences4table<-c(recurrences4table,recurrence)
                      
                      # Count spikes for energy estimation
                      spike_count <- count_spikes(
                        paste0(dataset_name, "/", dataset),
                        nu1, nu2, n_neuron, threshold, decay, amplitude, 
                        epochs, resolution, recurrence, fold_ids
                      )
                      spike_counts <- c(spike_counts, spike_count)
                    }
                    
                  } else {
                    # For non-recurrent networks, nu2 is not applicable
                     datasets4table<-c(datasets4table,dataset)
                     nu1s4table<-c(nu1s4table,nu1)
                     nu2s4table<-c(nu2s4table,NA)
                     ns4table<-c(ns4table,n_neuron)
                     thresholds4table<-c(thresholds4table,threshold)
                     decays4table<-c(decays4table,decay)
                     amplitudes4table<-c(amplitudes4table,amplitude)
                     epochss4table<-c(epochss4table,epochs)
                     resolutions4table<-c(resolutions4table,resolution)
                     recurrences4table<-c(recurrences4table,recurrence)
                    
                    # Count spikes for energy estimation (nu2 set to default)
                    spike_count <- count_spikes(
                      paste0(dataset_name, "/", dataset),
                      nu1, "0.1_ -0.1", n_neuron, threshold, decay, amplitude,
                      epochs, resolution, recurrence, fold_ids
                    )
                    spike_counts <- c(spike_counts, spike_count)
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

# Create energy consumption table with collected data
energy_table <- data.frame(
  dataset = datasets4table,
  nu1 = nu1s4table,
  nu2 = nu2s4table,
  n = ns4table,
  threshold = thresholds4table,
  decay = decays4table,
  amplitude = amplitudes4table,
  epochs = epochss4table,
  resolution = resolutions4table,
  recurrent = recurrences4table,
  spike_counts = spike_counts
)



