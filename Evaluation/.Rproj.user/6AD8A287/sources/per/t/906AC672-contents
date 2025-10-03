###############################################################################
# Main entry point to aggregate evaluation tables for multiple model families.
# This script configures hyperparameters, iterates over dataset origins, and
# sources the specific table-generation scripts. Variable names are defined in
# English for clarity; legacy aliases are provided for backward compatibility
# with the sourced scripts that expect Spanish variable names.
###############################################################################

# ----- Global configuration (common) -----

# 5-fold identifiers expected by downstream scripts
fold_ids <- c("1","2","3","4","5")
main_datasets<-c("Dodgers","CalIt2","Numenta")

# ----- Vacuum Spiker algorithm configuration -----

#Compile energy measurement routines
source("dependencies/energy_measurement.R")

#Compile performance measurement routines
source("dependencies/performance_measurement.R")

#Models that have been employed as baselines

ann_models <- c(
  "AdaptiveZhengZhenyu",
  "AdaptiveCaiWenjuan",
  "AdaptiveOhShuLih",
  "YildirimOzal",
  "Conv1dAutoencoder",
  "LSTMAutoencoder"
)

ml_models<-c("lof","ocsvm")

# Hyperparameter grids for SNN models
nu1_values <- c("0.1_ 0.1","0.1_ -0.1","-0.1_ 0.1","-0.1_ -0.1")
nu2_values <- c("0.1_ 0.1","0.1_ -0.1","-0.1_ 0.1","-0.1_ -0.1")
neuron_counts <- c("100","2000")
threshold_values <- c("-40.0","-55.0","-62.0")
decay_values <- c("100.0","150.0","200.0")
amplitude_values <- c("1.0")
epoch_values_snn <- c("1","2","3","4","5")
resolution_values <- c("0.1","0.001")
recurrent_flags <- c("True","False")

# ----- Generate SNN tables -----

for (dataset_origin in main_datasets) {
  # Path that generate_table_snn.R will use to discover datasets
  dataset_name <- paste0("output/Vacuum_Spiker/", dataset_origin)
  #Performance metrics
  source("search_codes/performance_search_snn.R")
  
  #Spike counts and MAC estimates
  source("search_codes/spikes_search_snn.R")
  energy_table<-estimate_snn_macs(energy_table)
  
  #Get together the information
  common_cols<-c("dataset","nu1","nu2","n","threshold",
                 "decay","amplitude","epochs","resolution",
                 "recurrent")
  
  tabla<-merge(energy_table,performance_table,by=common_cols,all.y=TRUE)
  
  write.table(tabla,paste0("tables/",dataset_origin,"_snn.csv"),quote=FALSE,
              sep=",",row.names=FALSE)
}



# ----- Deep Learning (ANN) configuration -----

batch_size_list <- c(32, 64, 128)
learning_rate_list <- c(0.001, 0.005, 0.01, 0.05, 0.1)
epoch_list <- c(10, 50, 100)

# ----- Generate ANN tables (per model and dataset origin) -----

for (dataset_origin in main_datasets) {
  tabla<-list()
  for (current_model in ann_models) {
    # Default per-model configuration
    n_layer_values <- c(1)

    # Sequence length depends on the model architecture
    if (current_model == "AdaptiveCaiWenjuan") {
      length_values <- c(67)
    } else if (current_model == "AdaptiveOhShuLih") {
      length_values <- c(20)
    } else if (current_model == "AdaptiveZhengZhenyu") {
      length_values <- c(256)
    } else {
      length_values <- c(50, 100, 150, 200)
    }

    # Adjust number of layers for certain models
    if (current_model == "Conv1dAutoencoder") {
      n_layer_values <- c(1, 2, 3)
    }

    # Hidden/latent sizes for LSTM autoencoder; defaults for others
    if (current_model == "LSTMAutoencoder") {
      n_layer_values <- c(1, 2, 3)
      hidden_values <- c(32, 64)
      latent_values <- c(20, 50)
    } else {
      hidden_values <- c(32)
      latent_values <- c(20)
    }

    # Path that generate_table_ann.R will use
    dataset_name <- paste0("output/DLBaselines/", current_model, "/", dataset_origin)
    source("search_codes/performance_search_ann.R")
  }
  
  tabla2write<-data.frame()
  
  for(t in tabla){tabla2write<-rbind(tabla2write,t)}
  
  #Add MAC estimates
  tabla2write<-estimate_ann_macs(tabla2write)
  
  write.table(
    tabla2write,
    paste0("tables/", dataset_origin, "_ann.csv"),
    quote = FALSE,
    sep = ",",
    row.names = FALSE
  )
}

# ----- Classic ML baselines (LOF, OCSVM) -----

for (dataset_origin in main_datasets) {
  # Execution of the model-specific grids
  source("search_codes/performance_search_ml.R")

  #Force the same columns in lof and ocsvm tables
  tabla_lof$nu<-(-1)
  tabla_lof$n_vectors<-(-1)
  tabla_ocsvm$neighbours<-(-1)
  
  #Get a single table for ml methods
  tabla<-rbind(tabla_lof,tabla_ocsvm)
  
  #Add MAC counts
  tabla<-estimate_ml_macs(tabla)
  
  write.table(tabla,paste0("tables/",dataset_origin,"_ml.csv"),quote=FALSE,sep=",",row.names=FALSE)
}

#Select the maximum value for each model and each metric, and the corresponding MACs.

for(dataset_origin in main_datasets){
  
  for(metric_sel in c("gm","auc","f1")){
    source("search_codes/search_best_results.R")
  }
}

#Apply statistical test to the obtained results:
for(metric in c("gm","f1","auc")){
  
  source("statistical_tests/global_tests.R")
  
}

#Finally, perform the Chi-Squared test
source("statistical_tests/synanptic_behaviour_study.R")




