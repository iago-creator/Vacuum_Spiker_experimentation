#Code to select the best results obtained for each dataset, metric and model.

# --- Libraries ---------------------------------------------------------------
library(dplyr)
library(purrr)

# --- Helpers -----------------------------------------------------------------
filter_by_condition <- function(data_list, condition) {
  # Filter every data frame in a named list using an R expression
  # provided as a string.
  map(data_list, ~ .x %>% filter(eval(parse(text = condition))))
}

# --- Input paths -------------------------------------------------------------
path <- "tables"

# --- Load & organize input data ---------------------------------------------
data <- list()

ann_df <- read.csv(paste0(path, "/", dataset_origin, "_ann.csv"))
for (m in ann_models) {
  data[[m]] <- ann_df[ann_df$model == m, ]
}

ml_df <- read.csv(paste0(path, "/", dataset_origin, "_ml.csv"))
for (m in ml_models) {
  data[[m]] <- ml_df[ml_df$model == m, ]
}

data$snn <- read.csv(paste0(path, "/", dataset_origin, "_snn.csv"))

# --- Core summary function ---------------------------------------------------
get_best_conf_by_metric <- function(data_list, metric) {
  # For each model and each dataset:
  #   - take the maximum value of the chosen metric
  #   - among rows achieving that max, take the minimum MACs (tie-breaker)
  # Return a wide data frame of best metric & MACs per model, plus the
  # parameter rows corresponding to those best settings.

  # Collect all dataset names present across inputs
  all_datasets <- character(0)
  for (nm in names(data_list)) {
    all_datasets <- unique(c(all_datasets, data_list[[nm]]$dataset))
  }

  # Prepare containers
  best_metrics <- list()
  best_macs    <- list()
  model_params <- list()

  # Ensure every model has a params data frame initialized
  for (nm in names(data_list)) model_params[[nm]] <- data.frame()

  # Clean and compute per model
  for (nm in names(data_list)) {
    df <- data_list[[nm]]

    # Drop NA/Inf in the selected metric
    df <- df[!is.infinite(df[[metric]]) & !is.na(df[[metric]]),]

    # Initialize per-model result vectors
    best_metrics[[nm]] <- c()
    best_macs[[nm]]    <- c()

    # Iterate over datasets
    for (d in all_datasets) {
      df_d <- df[df$dataset == d,]
      
	  # Best metric within this dataset
	  best_metric_value <- max(df_d[[metric]])
	  df_best <- df_d[df_d[[metric]] == best_metric_value,]
      # Among rows with best metric, pick the one with lowest MACs
	  best_mac_value <- min(df_best$macs)
      if (nrow(df_d) > 0) {
        #Select the corresponding parameters
        model_params[[nm]] <- rbind(model_params[[nm]], df_best[df_best$macs==best_mac_value,][1, ])
      }
        # Append results
        best_metrics[[nm]] <- c(best_metrics[[nm]], best_metric_value)
        best_macs[[nm]]    <- c(best_macs[[nm]], best_mac_value)
      }
    }
  
  # Build wide summary table
    summary_df <- data.frame(
      dataset = all_datasets,
      performance_snn = best_metrics$snn, macs_snn = best_macs$snn,
      performance_YildirimOzal = best_metrics$YildirimOzal, macs_YildirimOzal = best_macs$YildirimOzal,
      performance_OhShuLih = best_metrics$AdaptiveOhShuLih, macs_OhShuLih = best_macs$AdaptiveOhShuLih,
      performance_CaiWenjuan = best_metrics$AdaptiveCaiWenjuan, macs_CaiWenjuan = best_macs$AdaptiveCaiWenjuan,
      performance_Conv1dAutoencoder = best_metrics$Conv1dAutoencoder, macs_Conv1dAutoencoder = best_macs$Conv1dAutoencoder,
      performance_LSTMAutoencoder = best_metrics$LSTMAutoencoder, macs_LSTMAutoencoder = best_macs$LSTMAutoencoder,
      performance_lof = best_metrics$lof, macs_lof = best_macs$lof,
      performance_ocsvm = best_metrics$ocsvm, macs_ocsvm = best_macs$ocsvm
    )

  list(data = summary_df, pars = model_params)
}

# --- Run summary --------------------------------------------
result <- get_best_conf_by_metric(data, metric_sel)

# --- Output directories ------------------------------------------------------
dir.create(paste0("results/", dataset_origin, "/", metric_sel, "/parameters"), recursive = TRUE)

# --- Write outputs ---------------------------------------------
write.table(
  result$data,
  paste0("results/", dataset_origin, "/", metric_sel, "/comparison.csv"),
  quote = FALSE, row.names = FALSE, sep = ","
)

for (m in names(result$pars)) {
  write.table(
    result$pars[[m]],
    paste0("results/", dataset_origin, "/", metric_sel, "/parameters/", m, ".csv"),
    quote = FALSE, row.names = FALSE, sep = ","
  )
}