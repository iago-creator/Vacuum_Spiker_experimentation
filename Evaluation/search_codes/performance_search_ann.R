###############################################################################
# Main script to generate ANN aggregation tables
# - Reads per-dataset experiment outputs under `dataset_name`
# - Iterates over model hyperparameters and fold ids
# - Collects metrics returned by `medir_results_ann` into a single data.frame
# - Writes a CSV into `tables/<dataset_origin>_ann_<current_model>.csv`
###############################################################################

# Discover datasets inside the configured dataset_name path
datasets <- list.files(dataset_name)

dataset4table <- c()
batch_size4table <- c()
learning_rate4table <- c()
epochs4table <- c()
model4table <- c()
length4table <- c()

averages4table <- c()        # moving-average window size used
gmeans4table <- c()
f1s4table <- c()
sensitivities4table <- c()
specificities4table <- c()
precisions4table <- c()
recalls4table <- c()
aucs4table <- c()
n_layers4table <- c()

hiddens4table <- c()
latents4table <- c()

for (dataset in datasets) {
  print(paste0("dataset: ", dataset))

  for (batch_size in batch_size_list) {
    print(paste0("batch_size: ", batch_size))

    for (learning_rate in learning_rate_list) {
      print(paste0("learning_rate: ", learning_rate))

      for (epoch in epoch_list) {
        print(paste0("epochs: ", epoch))

        for (seq_length in length_values) {
          print(paste0("sequence_length: ", seq_length))

          for (num_layers in n_layer_values) {
            print(paste0("num_layers: ", num_layers))

            for (hidden_size in hidden_values) {
              print(paste0("hidden_size: ", hidden_size))

              for (latent_size in latent_values) {
                print(paste0("latent_size: ", latent_size))

                # Accumulate repeated descriptors per each metric variant (4 averages)
                dataset4table <- c(dataset4table, rep(dataset, 4))
                batch_size4table <- c(batch_size4table, rep(batch_size, 4))
                learning_rate4table <- c(learning_rate4table, rep(learning_rate, 4))
                epochs4table <- c(epochs4table, rep(epoch, 4))
                model4table <- c(model4table, rep(current_model, 4))
                length4table <- c(length4table, rep(seq_length, 4))
                n_layers4table <- c(n_layers4table, rep(num_layers, 4))
                hiddens4table <- c(hiddens4table, rep(hidden_size, 4))
                latents4table <- c(latents4table, rep(latent_size, 4))

                # Measure metrics across folds for this configuration
                r <- measure_performance_ann(
                  paste0(dataset_name, "/", datasets),
                  seq_length,
                  batch_size,
                  learning_rate,
                  epoch,
                  fold_ids,
                  current_model,
                  num_layers,
                  hidden_size,
                  latent_size
                )

                averages4table <- c(averages4table, r$average)
                gmeans4table <- c(gmeans4table, r$gm)
                f1s4table <- c(f1s4table, r$f1)
                sensitivities4table <- c(sensitivities4table, r$sensitivity)
                specificities4table <- c(specificities4table, r$specificity)
                precisions4table <- c(precisions4table, r$precision)
                recalls4table <- c(recalls4table, r$recall)
                aucs4table <- c(aucs4table, r$auc)
              }
            }
          }
        }
      }
    }
  }
}
tabla[[current_model]] <- data.frame(
  dataset = dataset4table,
  batch_size = batch_size4table,
  learning_rate = learning_rate4table,
  epochs = epochs4table,
  model = model4table,
  average = averages4table,
  gm = gmeans4table,
  f1 = f1s4table,
  sensitivity = sensitivities4table,
  specificity = specificities4table,
  precision = precisions4table,
  recall = recalls4table,
  auc = aucs4table,
  length = length4table,
  latent = latents4table,
  hidden = hiddens4table,
  n_layers = n_layers4table
)
