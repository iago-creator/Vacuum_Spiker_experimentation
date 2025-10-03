###############################################################################
# Utilities for computing performance metrics in time-series anomaly detection.
# 
# This module loads per-fold labels and score/spike traces from experiment
# directories (organized by hyperparameters), smooths them with moving-average
# windows (1, 100, 200, 300), sweeps decision thresholds across the score range,
# builds confusion matrices (caret) and derives GM, F1, Sensitivity, Specificity,
# Precision/Recall, and AUC (pROC). Metrics are aggregated across folds for each
# window. For OCSVM, it also reports the average number of support vectors.
#
# Dependencies: pracma (movavg), pROC (auc), caret (confusionMatrix).
#
# Outputs (per window): average, gm, f1, sensitivity, specificity, precision,
# recall, auc; plus n_vectors for OCSVM.
###############################################################################


###############################################################################
# Metrics computation for SNN models
# - Smooths spike traces per fold with moving averages
# - Sweeps thresholds to compute confusion matrices and metrics
# - Aggregates per-average-window metrics across folds
###############################################################################

# Moving average
library(pracma)

# ROC AUC
library(pROC)

# Confusion matrix and class metrics
library(caret)

measure_performance_snn <- function(dataset_root, nu1, nu2, n_neuron, threshold, decay, amplitude, epochs, resolution, recurrence, fold_ids) {
  
  # If non-recurrent, nu2 is irrelevant (use default value)
  if (recurrence == "False") { nu2 <- "0.1_ -0.1" }
  
  # Moving-average windows to evaluate
  average_windows <- c(1, 100, 200, 300)
  
  # Build path to experiment outputs
  path <- paste(dataset_root, nu1, nu2, n_neuron, threshold, decay, amplitude, epochs, resolution, recurrence, sep = "/")
  
  labels_by_fold <- list()
  spikes_by_fold <- list()
  
  # Read experiment outputs for each fold
  for (fold in fold_ids) {
    labels_by_fold[[fold]] <- read.csv(paste0(path, "/label_", fold), header = FALSE)$V1
    spikes_by_fold[[fold]] <- read.csv(paste0(path, "/spikes_", fold), header = FALSE)$V1
    
    # Clean labels: remove NaN and convert 0s to NA
    labels_by_fold[[fold]] <- labels_by_fold[[fold]][!is.nan(labels_by_fold[[fold]])]
    
    # Align spikes to label length
    spikes_by_fold[[fold]] <- spikes_by_fold[[fold]][1:length(labels_by_fold[[fold]])]
  }
  
  # Initialize aggregation containers
  gms <- c(); f1s <- c(); aucs <- c()
  sensitivities <- c(); specificities <- c(); precisions <- c(); recalls <- c()
  averages <- c()
  
  # For each moving-average window, compute metrics across folds
  for (avg_window in average_windows) {
    
    smoothed_by_fold <- list()
    skip_avg <- 0
    
    for (fold in fold_ids) {
      if (avg_window > 1) {
        if (avg_window < length(spikes_by_fold[[fold]])) {
          smoothed_by_fold[[fold]] <- movavg(spikes_by_fold[[fold]], avg_window)
        } else {
          skip_avg <- 1
        }
      } else {
        smoothed_by_fold[[fold]] <- spikes_by_fold[[fold]]
      }
    }
    
    invalid_folds <- 0
    gm_fold <- c(); f1_fold <- c(); sens_fold <- c(); spec_fold <- c()
    prec_fold <- c(); recall_fold <- c(); auc_fold <- c()
    
    # Iterate over folds and collect metrics
    for (fold in fold_ids) {
      
      scores <- smoothed_by_fold[[fold]]
      labs <- labels_by_fold[[fold]]
      labs[is.na(labs)] <- 0
      
      # Align and clean data
      labs <- labs[1:length(scores)]
      labs <- labs[!is.na(scores)]; scores <- scores[!is.na(scores)]
      
      # Check if evaluation is possible
      if ((length(unique(labs)) == 1) || (length(labs) == 0) || (skip_avg == 1)) {
        invalid_folds <- invalid_folds + 1
      } else {
        if (length(unique(labs[!is.na(labs)])) == 2) {
          
          # Sweep thresholds to find optimal operating points
          thresholds <- seq(min(scores), max(scores), (max(scores) - min(scores)) / 10)
          
          gm_tmp <- c(); f1_tmp <- c(); sens_tmp <- c(); spec_tmp <- c()
          prec_tmp <- c(); rec_tmp <- c()
          
          for (th in thresholds) {
            scores_bin <- scores
            scores_bin[scores_bin <= th] <- 0
            scores_bin[scores_bin > 0] <- 1
            
            scores_bin <- factor(scores_bin, levels = c(0, 1))
            labs_bin <- factor(labs, levels = c(0, 1))
            
            cm <- confusionMatrix(scores_bin, labs_bin, positive = "1")
            
            gm_val <- sqrt(cm$byClass["Sensitivity"] * cm$byClass["Specificity"])
            gm_tmp <- c(gm_tmp, gm_val)
            
            f1_val <- if (!is.na(gm_val) && (is.na(cm$byClass["F1"]))) 0 else cm$byClass["F1"]
            f1_tmp <- c(f1_tmp, f1_val)
            
            sens_tmp <- c(sens_tmp, cm$byClass["Sensitivity"])
            spec_tmp <- c(spec_tmp, cm$byClass["Specificity"])
            prec_tmp <- c(prec_tmp, cm$byClass["Precision"])
            rec_tmp <- c(rec_tmp, cm$byClass["Recall"])
          }
          
          # Select best metrics based on GM and F1 optimization
          gm_fold <- c(gm_fold, max(gm_tmp[!is.na(gm_tmp)]))
          f1_fold <- c(f1_fold, max(f1_tmp[!is.na(f1_tmp)]))
          
          # Get corresponding sensitivity/specificity at best GM
          best_gm_idx <- which(gm_tmp == max(gm_tmp[!is.na(gm_tmp)]) & !is.na(gm_tmp))[1]
          sens_fold <- c(sens_fold, sens_tmp[best_gm_idx])
          spec_fold <- c(spec_fold, spec_tmp[best_gm_idx])
          
          # Get corresponding precision/recall at best F1
          best_f1_idx <- which(f1_tmp == max(f1_tmp[!is.na(f1_tmp)]) & !is.na(f1_tmp))[1]
          prec_fold <- c(prec_fold, prec_tmp[best_f1_idx])
          recall_fold <- c(recall_fold, rec_tmp[best_f1_idx])
          
          # Compute AUC
          auc_fold <- c(auc_fold, auc(labs, scores, direction = "<", levels = c("0", "1")))
        }
      }
    }
    
    # Aggregate metrics across folds or mark as failed
    if (is.null(auc_fold)) { invalid_folds <- 5 }
    
    if (invalid_folds == 5) {
      gms <- c(gms, -1); f1s <- c(f1s, -1)
      sensitivities <- c(sensitivities, -1); specificities <- c(specificities, -1)
      precisions <- c(precisions, -1); recalls <- c(recalls, -1); aucs <- c(aucs, -1)
      averages <- c(averages, avg_window)
    } else {
      gms <- c(gms, mean(gm_fold)); f1s <- c(f1s, mean(f1_fold)); aucs <- c(aucs, mean(auc_fold))
      sensitivities <- c(sensitivities, mean(sens_fold)); specificities <- c(specificities, mean(spec_fold))
      precisions <- c(precisions, mean(prec_fold)); recalls <- c(recalls, mean(recall_fold))
      averages <- c(averages, avg_window)
    }
  }
  
  return(list(
    average = averages,
    gm = gms,
    f1 = f1s,
    sensitivity = sensitivities,
    specificity = specificities,
    precision = precisions,
    recall = recalls,
    auc = aucs
  ))
}


###############################################################################
# Metrics computation for ANN models
# - Smooths score traces per fold with moving averages
# - Sweeps thresholds to compute confusion matrices and derived metrics
# - Aggregates per-average-window metrics across folds
###############################################################################

# Moving average
library(pracma)

# ROC AUC
library(pROC)

# Confusion matrix and class metrics
library(caret)

measure_performance_ann <- function(dataset_root,
                              sequence_length,
                              batch_size,
                              learning_rate,
                              epochs,
                              fold_ids,
                              model_name,
                              num_layers,
                              hidden_size,
                              latent_size){
  #browser()
  # Build configuration path according to model layout
  path <- paste(dataset_root, sequence_length, batch_size, learning_rate, epochs, sep = "/")
  if (model_name == "Conv1dAutoencoder"){
    path <- paste(path, "None/None", num_layers, sep = "/")
  } else if (model_name == "LSTMAutoencoder"){
    path <- paste(path, hidden_size, latent_size, num_layers, sep = "/")
  }
  
  average_windows <- c(1, 100, 200, 300)
  labels_by_fold <- list()
  traces_by_fold <- list()
  
  # Validate required files for all folds
  if(!file.exists(paste0(path, "/label_1")) || !file.exists(paste0(path, "/label_2")) ||
     !file.exists(paste0(path, "/label_3")) || !file.exists(paste0(path, "/label_4")) ||
     !file.exists(paste0(path, "/label_5"))){
    averages <- c()
    for (w in average_windows){
      averages <- c(averages, w)
    }
    return(list(
      average = averages,
      gm = rep(-1, length(average_windows)),
      f1 = rep(-1, length(average_windows)),
      sensitivity = rep(-1, length(average_windows)),
      specificity = rep(-1, length(average_windows)),
      precision = rep(-1, length(average_windows)),
      recall = rep(-1, length(average_windows)),
      auc = rep(-1, length(average_windows))
    ))
  }
  
  for (fold in fold_ids){
    if (!file.size(paste0(path, "/label_", fold)) == 0){
      labels_by_fold[[fold]] <- read.csv(paste0(path, "/label_", fold), header = FALSE)$V1
    }
    if (!file.size(paste0(path, "/traza_", fold)) == 0){
      traces_by_fold[[fold]] <- read.csv(paste0(path, "/traza_", fold), header = FALSE)$V1
    }
  }
  
  gms <- c(); f1s <- c(); aucs <- c();
  sensitivities <- c(); specificities <- c(); precisions <- c(); recalls <- c();
  averages <- c()
  
  # For each moving-average window, compute metrics across folds
  for (avg_window in average_windows){
    smoothed_by_fold <- list()
    skip_avg <- 0
    for (fold in fold_ids){
      if (avg_window > 1){
        if (avg_window < length(traces_by_fold[[fold]])){
          smoothed_by_fold[[fold]] <- movavg(traces_by_fold[[fold]], avg_window)
        } else {
          skip_avg <- 1
        }
      } else {
        smoothed_by_fold[[fold]] <- traces_by_fold[[fold]]
      }
    }
    
    invalid_folds <- 0
    gm_fold <- c(); f1_fold <- c(); sens_fold <- c(); spec_fold <- c();
    prec_fold <- c(); recall_fold <- c(); auc_fold <- c()
    
    for (fold in fold_ids){
      scores <- smoothed_by_fold[[fold]]
      labs <- labels_by_fold[[fold]]
      labs[is.na(labs)] <- 0
      
      # Align and clean
      labs <- labs[1:length(scores)]
      labs <- labs[!is.na(scores)]; scores <- scores[!is.na(scores)]
      labs <- labs[!is.infinite(scores)]; scores <- scores[!is.infinite(scores)]
      labs <- labs[!is.nan(scores)]; scores <- scores[!is.nan(scores)]
      
	  # Check if evaluation is possible
      if ((length(unique(labs))==1) || (length(labs)==0) || (skip_avg==1)){
        invalid_folds <- invalid_folds + 1
      } else {
        if (length(unique(labs[!is.na(labs)]))==2){
		  
		  # Sweep thresholds to find optimal operating points
          thresholds <- seq(min(scores), max(scores), (max(scores)-min(scores))/10)
          gm_tmp <- c(); f1_tmp <- c(); sens_tmp <- c(); spec_tmp <- c(); prec_tmp <- c(); rec_tmp <- c()
          for (th in thresholds){
            sc <- scores
            sc[sc<=th] <- 0; sc[sc>0] <- 1
            sc <- factor(sc, levels=c(0,1))
            lb <- factor(labs, levels=c(0,1))
            cm <- confusionMatrix(sc, lb, positive="1")
            gm_val <- sqrt(cm$byClass["Sensitivity"]*cm$byClass["Specificity"])
            gm_tmp <- c(gm_tmp, gm_val)
            f1_val <- if (!is.na(gm_val) && (is.na(cm$byClass["F1"]))) 0 else cm$byClass["F1"]
            f1_tmp <- c(f1_tmp, f1_val)
            sens_tmp <- c(sens_tmp, cm$byClass["Sensitivity"])
            spec_tmp <- c(spec_tmp, cm$byClass["Specificity"])
            prec_tmp <- c(prec_tmp, cm$byClass["Precision"])
            rec_tmp <- c(rec_tmp, cm$byClass["Recall"])
          }
		  
		  # Select best metrics based on GM and F1 optimization
          gm_fold <- c(gm_fold, max(gm_tmp[!is.na(gm_tmp)]))
          f1_fold <- c(f1_fold, max(f1_tmp[!is.na(f1_tmp)]))
		  
		   # Get corresponding sensitivity/specificity at best GM
		  best_gm_idx <- which(gm_tmp == max(gm_tmp[!is.na(gm_tmp)]) & !is.na(gm_tmp))[1]
          sens_fold <- c(sens_fold, sens_tmp[best_gm_idx])
          spec_fold <- c(spec_fold, spec_tmp[best_gm_idx])
          
		  # Get corresponding precision/recall at best F1
		  best_f1_idx <- which(f1_tmp == max(f1_tmp[!is.na(f1_tmp)]) & !is.na(f1_tmp))[1]
		  prec_fold <- c(prec_fold, prec_tmp[best_f1_idx])
          recall_fold <- c(recall_fold, rec_tmp[best_f1_idx])
          
		  # Compute AUC
		  auc_fold <- c(auc_fold, auc(labs, scores, direction="<", levels=c("0","1")))
        }
      }
    }
    
	# Aggregate metrics across folds or mark as failed
    if (is.null(auc_fold)) { invalid_folds <- 5 }
    if (invalid_folds == 5){
      gms <- c(gms, -1); f1s <- c(f1s, -1)
      sensitivities <- c(sensitivities, -1); specificities <- c(specificities, -1)
      precisions <- c(precisions, -1); recalls <- c(recalls, -1); aucs <- c(aucs, -1)
      averages <- c(averages, avg_window)
    } else {
      gms <- c(gms, mean(gm_fold)); f1s <- c(f1s, mean(f1_fold)); aucs <- c(aucs, mean(auc_fold))
      sensitivities <- c(sensitivities, mean(sens_fold)); specificities <- c(specificities, mean(spec_fold))
      precisions <- c(precisions, mean(prec_fold)); recalls <- c(recalls, mean(recall_fold))
      averages <- c(averages, avg_window)
    }
  }
  
  return(list(
    average = averages,
    gm = gms,
    f1 = f1s,
    sensitivity = sensitivities,
    specificity = specificities,
    precision = precisions,
    recall = recalls,
    auc = aucs
  ))
}


###############################################################################
# Metrics computation for LOF baseline
# - Smooths score traces per fold with moving averages
# - Sweeps thresholds to compute confusion matrices and metrics
# - Aggregates per-average-window metrics across folds
###############################################################################

# Moving average
library(pracma)

# ROC AUC
library(pROC)

# Confusion matrix and class metrics
library(caret)

measure_performance_lof <- function(dataset_root, sequence_length, neighbors_count, fold_ids) {
  
  # Build path to experiment outputs
  path <- paste(dataset_root, sequence_length, neighbors_count, sep = "/")
  
  # Moving-average windows to evaluate
  average_windows <- c(1, 100, 200, 300)
  
  labels_by_fold <- list()
  scores_by_fold <- list()

  # Validate required files exist for all folds
  if (!file.exists(paste0(path, "/label_1")) || !file.exists(paste0(path, "/label_2")) ||
      !file.exists(paste0(path, "/label_3")) || !file.exists(paste0(path, "/label_4")) ||
      !file.exists(paste0(path, "/label_5"))) {
    
    averages <- c()
    for (w in average_windows) {
      averages <- c(averages, w)
    }
    
    return(list(
      average = averages,
      gm = rep(-1, length(average_windows)),
      f1 = rep(-1, length(average_windows)),
      sensitivity = rep(-1, length(average_windows)),
      specificity = rep(-1, length(average_windows)),
      precision = rep(-1, length(average_windows)),
      recall = rep(-1, length(average_windows)),
      auc = rep(-1, length(average_windows))
    ))
  }

  # Read experiment outputs for each fold
  for (fold in fold_ids) {
    if (!file.size(paste0(path, "/label_", fold)) == 0) {
      labels_by_fold[[fold]] <- read.csv(paste0(path, "/label_", fold), header = FALSE)$V1
    }
    if (!file.size(paste0(path, "/traza_", fold)) == 0) {
      scores_by_fold[[fold]] <- read.csv(paste0(path, "/traza_", fold), header = FALSE)$V1
    }
  }

  # Initialize aggregation containers
  gms <- c(); f1s <- c(); aucs <- c()
  sensitivities <- c(); specificities <- c(); precisions <- c(); recalls <- c()
  averages <- c()

  # For each moving-average window, compute metrics across folds
  for (avg_window in average_windows) {
    
    smoothed_by_fold <- list()
    skip_avg <- 0
    
    for (fold in fold_ids) {
      if (avg_window > 1) {
        if (avg_window < length(scores_by_fold[[fold]])) {
          smoothed_by_fold[[fold]] <- movavg(scores_by_fold[[fold]], avg_window)
        } else {
          skip_avg <- 1
        }
      } else {
        smoothed_by_fold[[fold]] <- scores_by_fold[[fold]]
      }
    }
    
    invalid_folds <- 0
    gm_fold <- c(); f1_fold <- c(); sens_fold <- c(); spec_fold <- c()
    prec_fold <- c(); recall_fold <- c(); auc_fold <- c()
    
    for (fold in fold_ids) {
      
      scores <- smoothed_by_fold[[fold]]
      labs <- labels_by_fold[[fold]]
      labs[is.na(labs)] <- 0
      
      # Align and clean data
      labs <- labs[1:length(scores)]
      labs <- labs[!is.na(scores)]; scores <- scores[!is.na(scores)]
      labs <- labs[!is.infinite(scores)]; scores <- scores[!is.infinite(scores)]
      labs <- labs[!is.nan(scores)]; scores <- scores[!is.nan(scores)]
      
      # Check if evaluation is possible
      if ((length(unique(labs)) == 1) || (length(labs) == 0) || (skip_avg == 1)) {
        invalid_folds <- invalid_folds + 1
      } else {
        if (length(unique(labs[!is.na(labs)])) == 2) {
          
          # Sweep thresholds to find optimal operating points
          thresholds <- seq(min(scores), max(scores), (max(scores) - min(scores)) / 10)
          
          gm_tmp <- c(); f1_tmp <- c(); sens_tmp <- c(); spec_tmp <- c()
          prec_tmp <- c(); rec_tmp <- c()
          
          for (th in thresholds) {
            scores_bin <- scores
            scores_bin[scores_bin <= th] <- 0
            scores_bin[scores_bin > 0] <- 1
            
            scores_bin <- factor(scores_bin, levels = c(0, 1))
            labs_bin <- factor(labs, levels = c(0, 1))
            
            cm <- confusionMatrix(scores_bin, labs_bin, positive = "1")
            
            gm_val <- sqrt(cm$byClass["Sensitivity"] * cm$byClass["Specificity"])
            gm_tmp <- c(gm_tmp, gm_val)
            
            f1_val <- if (!is.na(gm_val) && (is.na(cm$byClass["F1"]))) 0 else cm$byClass["F1"]
            f1_tmp <- c(f1_tmp, f1_val)
            
            sens_tmp <- c(sens_tmp, cm$byClass["Sensitivity"])
            spec_tmp <- c(spec_tmp, cm$byClass["Specificity"])
            prec_tmp <- c(prec_tmp, cm$byClass["Precision"])
            rec_tmp <- c(rec_tmp, cm$byClass["Recall"])
          }
          
          # Select best metrics based on GM and F1 optimization
          gm_fold <- c(gm_fold, max(gm_tmp[!is.na(gm_tmp)]))
          f1_fold <- c(f1_fold, max(f1_tmp[!is.na(f1_tmp)]))
          
          # Get corresponding sensitivity/specificity at best GM
          best_gm_idx <- which(gm_tmp == max(gm_tmp[!is.na(gm_tmp)]) & !is.na(gm_tmp))[1]
          sens_fold <- c(sens_fold, sens_tmp[best_gm_idx])
          spec_fold <- c(spec_fold, spec_tmp[best_gm_idx])
          
          # Get corresponding precision/recall at best F1
          best_f1_idx <- which(f1_tmp == max(f1_tmp[!is.na(f1_tmp)]) & !is.na(f1_tmp))[1]
          prec_fold <- c(prec_fold, prec_tmp[best_f1_idx])
          recall_fold <- c(recall_fold, rec_tmp[best_f1_idx])
          
          # Compute AUC
          auc_fold <- c(auc_fold, auc(labs, scores, direction = "<", levels = c("0", "1")))
        }
      }
    }

    # Aggregate metrics across folds or mark as failed
    if (is.null(auc_fold)) { invalid_folds <- 5 }
    
    if (invalid_folds == 5) {
      gms <- c(gms, -1); f1s <- c(f1s, -1)
      sensitivities <- c(sensitivities, -1); specificities <- c(specificities, -1)
      precisions <- c(precisions, -1); recalls <- c(recalls, -1); aucs <- c(aucs, -1)
      averages <- c(averages, avg_window)
    } else {
      gms <- c(gms, mean(gm_fold)); f1s <- c(f1s, mean(f1_fold)); aucs <- c(aucs, mean(auc_fold))
      sensitivities <- c(sensitivities, mean(sens_fold)); specificities <- c(specificities, mean(spec_fold))
      precisions <- c(precisions, mean(prec_fold)); recalls <- c(recalls, mean(recall_fold))
      averages <- c(averages, avg_window)
    }
  }

  return(list(
    average = averages,
    gm = gms,
    f1 = f1s,
    sensitivity = sensitivities,
    specificity = specificities,
    precision = precisions,
    recall = recalls,
    auc = aucs
  ))
}


###############################################################################
# Metrics computation for OCSVM baseline
# - Smooths score traces per fold with moving averages
# - Sweeps thresholds to compute confusion matrices and metrics
# - Aggregates per-average-window metrics across folds
# - Returns support vector count information
###############################################################################

# Moving average
library(pracma)

# ROC AUC
library(pROC)

# Confusion matrix and class metrics
library(caret)

measure_performance_ocsvm <- function(dataset_root, sequence_length, nu_param, fold_ids) {
  
  # Build path to experiment outputs
  path <- paste(dataset_root, sequence_length, nu_param, sep = "/")
  
  # Moving-average windows to evaluate
  average_windows <- c(1, 100, 200, 300)
  
  labels_by_fold <- list()
  scores_by_fold <- list()
  support_vectors <- c()
  
  # Validate required files exist for all folds
  if (!file.exists(paste0(path, "/label_1")) || !file.exists(paste0(path, "/label_2")) ||
      !file.exists(paste0(path, "/label_3")) || !file.exists(paste0(path, "/label_4")) ||
      !file.exists(paste0(path, "/label_5"))) {
    
    averages <- c()
    for (w in average_windows) {
      averages <- c(averages, w)
    }
    
    return(list(
      average = averages,
      gm = rep(-1, length(average_windows)),
      f1 = rep(-1, length(average_windows)),
      sensitivity = rep(-1, length(average_windows)),
      specificity = rep(-1, length(average_windows)),
      precision = rep(-1, length(average_windows)),
      recall = rep(-1, length(average_windows)),
      auc = rep(-1, length(average_windows)),
      n_vectors = rep(-1, length(average_windows))
    ))
  }
  
  # Read experiment outputs for each fold
  for (fold in fold_ids) {
    if (!file.size(paste0(path, "/label_", fold)) == 0) {
      labels_by_fold[[fold]] <- read.csv(paste0(path, "/label_", fold), header = FALSE)$V1
    }
    if (!file.size(paste0(path, "/traza_", fold)) == 0) {
      scores_by_fold[[fold]] <- read.csv(paste0(path, "/traza_", fold), header = FALSE)$V1
    }
    if (!file.size(paste0(path, "/n_vectors_", fold)) == 0) {
      support_vectors <- c(support_vectors, scan(paste0(path, "/n_vectors_", fold)))
    }
  }
  
  # Initialize aggregation containers
  gms <- c(); f1s <- c(); aucs <- c()
  sensitivities <- c(); specificities <- c(); precisions <- c(); recalls <- c()
  averages <- c()
  
  # Average support vector count across folds
  n_vectors <- rep(mean(support_vectors), length(average_windows))

  # For each moving-average window, compute metrics across folds
  for (avg_window in average_windows) {
    
    smoothed_by_fold <- list()
    skip_avg <- 0
    
    for (fold in fold_ids) {
      if (avg_window > 1) {
        if (avg_window < length(scores_by_fold[[fold]])) {
          smoothed_by_fold[[fold]] <- movavg(scores_by_fold[[fold]], avg_window)
        } else {
          skip_avg <- 1
        }
      } else {
        smoothed_by_fold[[fold]] <- scores_by_fold[[fold]]
      }
    }
    
    invalid_folds <- 0
    gm_fold <- c(); f1_fold <- c(); sens_fold <- c(); spec_fold <- c()
    prec_fold <- c(); recall_fold <- c(); auc_fold <- c()
    
    for (fold in fold_ids) {
      
      scores <- smoothed_by_fold[[fold]]
      labs <- labels_by_fold[[fold]]
      labs[is.na(labs)] <- 0
      
      # Align and clean data
      labs <- labs[1:length(scores)]
      labs <- labs[!is.na(scores)]; scores <- scores[!is.na(scores)]
      labs <- labs[!is.infinite(scores)]; scores <- scores[!is.infinite(scores)]
      labs <- labs[!is.nan(scores)]; scores <- scores[!is.nan(scores)]
      
      # Check if evaluation is possible
      if ((length(unique(labs)) == 1) || (length(labs) == 0) || (skip_avg == 1)) {
        invalid_folds <- invalid_folds + 1
      } else {
        if (length(unique(labs[!is.na(labs)])) == 2) {
          
          # Sweep thresholds to find optimal operating points
          thresholds <- seq(min(scores), max(scores), (max(scores) - min(scores)) / 10)
          
          gm_tmp <- c(); f1_tmp <- c(); sens_tmp <- c(); spec_tmp <- c()
          prec_tmp <- c(); rec_tmp <- c()
          
          for (th in thresholds) {
            scores_bin <- scores
            scores_bin[scores_bin <= th] <- 0
            scores_bin[scores_bin > 0] <- 1
            
            scores_bin <- factor(scores_bin, levels = c(0, 1))
            labs_bin <- factor(labs, levels = c(0, 1))
            
            cm <- confusionMatrix(scores_bin, labs_bin, positive = "1")
            
            gm_val <- sqrt(cm$byClass["Sensitivity"] * cm$byClass["Specificity"])
            gm_tmp <- c(gm_tmp, gm_val)
            
            f1_val <- if (!is.na(gm_val) && (is.na(cm$byClass["F1"]))) 0 else cm$byClass["F1"]
            f1_tmp <- c(f1_tmp, f1_val)
            
            sens_tmp <- c(sens_tmp, cm$byClass["Sensitivity"])
            spec_tmp <- c(spec_tmp, cm$byClass["Specificity"])
            prec_tmp <- c(prec_tmp, cm$byClass["Precision"])
            rec_tmp <- c(rec_tmp, cm$byClass["Recall"])
          }
          
          # Select best metrics based on GM and F1 optimization
          gm_fold <- c(gm_fold, max(gm_tmp[!is.na(gm_tmp)]))
          f1_fold <- c(f1_fold, max(f1_tmp[!is.na(f1_tmp)]))
          
          # Get corresponding sensitivity/specificity at best GM
          best_gm_idx <- which(gm_tmp == max(gm_tmp[!is.na(gm_tmp)]) & !is.na(gm_tmp))[1]
          sens_fold <- c(sens_fold, sens_tmp[best_gm_idx])
          spec_fold <- c(spec_fold, spec_tmp[best_gm_idx])
          
          # Get corresponding precision/recall at best F1
          best_f1_idx <- which(f1_tmp == max(f1_tmp[!is.na(f1_tmp)]) & !is.na(f1_tmp))[1]
          prec_fold <- c(prec_fold, prec_tmp[best_f1_idx])
          recall_fold <- c(recall_fold, rec_tmp[best_f1_idx])
          
          # Compute AUC
          auc_fold <- c(auc_fold, auc(labs, scores, direction = "<", levels = c("0", "1")))
        }
      }
    }
    
    # Aggregate metrics across folds or mark as failed
    if (is.null(auc_fold)) { invalid_folds <- 5 }
    
    if (invalid_folds == 5) {
      gms <- c(gms, -1); f1s <- c(f1s, -1)
      sensitivities <- c(sensitivities, -1); specificities <- c(specificities, -1)
      precisions <- c(precisions, -1); recalls <- c(recalls, -1); aucs <- c(aucs, -1)
      averages <- c(averages, avg_window)
    } else {
      gms <- c(gms, mean(gm_fold)); f1s <- c(f1s, mean(f1_fold)); aucs <- c(aucs, mean(auc_fold))
      sensitivities <- c(sensitivities, mean(sens_fold)); specificities <- c(specificities, mean(spec_fold))
      precisions <- c(precisions, mean(prec_fold)); recalls <- c(recalls, mean(recall_fold))
      averages <- c(averages, avg_window)
    }
  }
  
  return(list(
    average = averages,
    gm = gms,
    f1 = f1s,
    sensitivity = sensitivities,
    specificity = specificities,
    precision = precisions,
    recall = recalls,
    auc = aucs,
    n_vectors = n_vectors
  ))
}


