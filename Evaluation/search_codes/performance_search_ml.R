###############################################################################
# Script to aggregate LOF and OCSVM baseline experiment results into summary tables.
#
# This script automates the collection and aggregation of performance metrics
# for classical anomaly detection baselines (LOF and OCSVM) across multiple
# datasets, hyperparameter configurations, and cross-validation folds.
#
# For each dataset and parameter combination:
# - Loads per-fold evaluation results computed by `measure_performance_lof()` or 
#   `measure_performance_ocsvm()`.
# - Iterates over predefined hyperparameter grids (sequence length, neighbours, nu).
# - Appends evaluation metrics (GM, F1, sensitivity, specificity, precision, 
#   recall, AUC, and number of support vectors for OCSVM) to cumulative lists.
# - Builds final aggregated `data.frame` objects (`tabla_lof` and `tabla_ocsvm`).


###############################################################################
# Main script to generate LOF aggregation tables
# - Reads per-dataset experiment outputs under `dataset_name`
# - Iterates over LOF hyperparameters and fold ids
# - Collects metrics returned by `measure_performance_lof` into a single data.frame
# - Writes a CSV into `tables/<dataset_origin>_ml_lof.csv`
###############################################################################

length_values <- c(50,100,150,200)
neighbours_values <- c(30,50)


current_model <- "lof"

dataset_name<-paste0("output/MLBaselines/",current_model,"/",dataset_origin)

datasets<-list.files(dataset_name)

dataset4table<-c()
model4table<-c()
length4table<-c()
neighbours4table<-c()

averages4table<-c()
gmeans4table<-c()
f1s4table<-c()
sensitivities4table<-c()
specificities4table<-c()
precisions4table<-c()
recalls4table<-c()
aucs4table<-c()

  
for(dataset in datasets){
  print(paste0("dataset: ",dataset))
  for (seq_length in length_values){
    print(paste0("sequence_length: ",seq_length))
    for (neighbours in neighbours_values){
      print(paste0("neighbours: ",neighbours))

      dataset4table<-c(dataset4table,rep(dataset,4))
      model4table<-c(model4table,rep(current_model,4))
      length4table<-c(length4table,rep(seq_length,4))
      neighbours4table<-c(neighbours4table,rep(neighbours,4))

      r<-measure_performance_lof(paste0(dataset_name,"/",datasets),seq_length,neighbours,fold_ids)

      averages4table<-c(averages4table,r$average)
      gmeans4table<-c(gmeans4table,r$gm)
      f1s4table<-c(f1s4table,r$f1)
      sensitivities4table<-c(sensitivities4table,r$sensitivity)
      specificities4table<-c(specificities4table,r$specificity)
      precisions4table<-c(precisions4table,r$precision)
      recalls4table<-c(recalls4table,r$recall)
      aucs4table<-c(aucs4table,r$auc)
    }
  }
}

tabla_lof<-data.frame(dataset=dataset4table,
                  model=model4table,
                  average=averages4table,
                  gm=gmeans4table,
                  f1=f1s4table,
                  sensitivity=sensitivities4table,
                  specificity=specificities4table,
                  precision=precisions4table,
                  recall=recalls4table,
                  auc=aucs4table,
                  length=length4table,
                  neighbours=neighbours4table
)


###############################################################################
# Main script to generate OCSVM aggregation tables
# - Reads per-dataset experiment outputs under `dataset_name`
# - Iterates over OCSVM hyperparameters and fold ids
# - Collects metrics returned by `medir_results_ocsvm` into a single data.frame
# - Writes a CSV into `tables/<dataset_origin>_ml_ocsvm.csv`
###############################################################################

length_values <- c(50,100,150,200)
nu_values <- c(0.05,0.2)


current_model <- "ocsvm"
dataset_name<-paste0("output/MLBaselines/",current_model,"/",dataset_origin)

datasets<-list.files(dataset_name)

dataset4table<-c()
model4table<-c()
length4table<-c()
nu4table<-c()

averages4table<-c()
gmeans4table<-c()
f1s4table<-c()
sensitivities4table<-c()
specificities4table<-c()
precisions4table<-c()
recalls4table<-c()
aucs4table<-c()
vectors4table<-c()

  
for(dataset in datasets){
    
  print(paste0("dataset: ",dataset))
  for (seq_length in length_values){
      print(paste0("sequence_length: ",seq_length))
      for (nu in nu_values){
          print(paste0("Nu:",nu))
          
          dataset4table<-c(dataset4table,rep(dataset,4))
          model4table<-c(model4table,rep(current_model,4))
          length4table<-c(length4table,rep(seq_length,4))
          nu4table<-c(nu4table,rep(nu,4))
          
          r<-measure_performance_ocsvm(paste0(dataset_name,"/",datasets),seq_length,nu,fold_ids)
          
          averages4table<-c(averages4table,r$average)
          gmeans4table<-c(gmeans4table,r$gm)
          f1s4table<-c(f1s4table,r$f1)
          sensitivities4table<-c(sensitivities4table,r$sensitivity)
          specificities4table<-c(specificities4table,r$specificity)
          precisions4table<-c(precisions4table,r$precision)
          recalls4table<-c(recalls4table,r$recall)
          aucs4table<-c(aucs4table,r$auc)
          vectors4table<-c(vectors4table,r$n_vectors)
      }
  }
}   

tabla_ocsvm<-data.frame(dataset=dataset4table,
                  model=model4table,
                  average=averages4table,
                  gm=gmeans4table,
                  f1=f1s4table,
                  sensitivity=sensitivities4table,
                  specificity=specificities4table,
                  precision=precisions4table,
                  recall=recalls4table,
                  auc=aucs4table,
                  length=length4table,
                  nu=nu4table,
                  n_vectors=vectors4table
)
