###############################################################################
# Main script to generate SNN aggregation tables
# - Reads per-dataset experiment outputs under `dataset_name`
# - Iterates over SNN hyperparameters and fold ids
# - Collects metrics returned by `measure_performance_snn` into a single data.frame
# - Writes a CSV into `tables/<dataset_origin>_snn.csv`
###############################################################################

# dataset_name is expected to be defined by the caller (ppal_general.R)

# Discover datasets inside the configured dataset_name path
datasets<-list.files(dataset_name)

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

averages4table<-c()
gmeans4table<-c()
f1s4table<-c()
sensitivities4table<-c()
specificities4table<-c()
precisions4table<-c()
recalls4table<-c()
aucs4table<-c()

for (dataset in datasets){
  for (nu1 in nu1_values){
    print(paste0("nu1: ",nu1))
    for (n in neuron_counts){
      print(paste0("n: ",n))
      for (threshold in threshold_values){
        print(paste0("threshold: ",threshold))
        for (decay in decay_values){
          print(paste0("decay: ",decay))
          for (amplitude in amplitude_values){
            print(paste0("amplitude: ",amplitude))
            for (epochs in epoch_values_snn){
              print(paste0("epochs: ",epochs))
              for (resolution in resolution_values){
                print(paste0("resolution: ",resolution))
                for(recurrence in recurrent_flags){
                  print(paste0("recurrence: ",recurrence))
                  if (recurrence=="True"){               
                    for (nu2 in nu2_values){
                      print(paste0("nu2: ",nu2))
                      
                      datasets4table<-c(datasets4table,rep(dataset,4))
                      nu1s4table<-c(nu1s4table,rep(nu1,4))
                      nu2s4table<-c(nu2s4table,rep(nu2,4))
                      ns4table<-c(ns4table,rep(n,4))
                      thresholds4table<-c(thresholds4table,rep(threshold,4))
                      decays4table<-c(decays4table,rep(decay,4))
                      amplitudes4table<-c(amplitudes4table,rep(amplitude,4))
                      epochss4table<-c(epochss4table,rep(epochs,4))
                      resolutions4table<-c(resolutions4table,rep(resolution,4))
                      recurrences4table<-c(recurrences4table,rep(recurrence,4))
                      r<-measure_performance_snn(paste0(dataset_name,"/",datasets),nu1,nu2,n,threshold,decay,amplitude,epochs,resolution,recurrence,fold_ids)
                      #Metemos resultados:
                  
                      averages4table<-c(averages4table,r$average)
                      gmeans4table<-c(gmeans4table,r$gm)
                      f1s4table<-c(f1s4table,r$f1)
                      sensitivities4table<-c(sensitivities4table,r$sensitivity)
                      specificities4table<-c(specificities4table,r$specificity)
                      precisions4table<-c(precisions4table,r$precision)
                      recalls4table<-c(recalls4table,r$recall)
                      aucs4table<-c(aucs4table,r$auc)
                      
                    }
                    
                  }else{
                    
                    datasets4table<-c(datasets4table,rep(dataset,4))
                    nu1s4table<-c(nu1s4table,rep(nu1,4))
                    nu2s4table<-c(nu2s4table,rep(NA,4))
                    ns4table<-c(ns4table,rep(n,4))
                    thresholds4table<-c(thresholds4table,rep(threshold,4))
                    decays4table<-c(decays4table,rep(decay,4))
                    amplitudes4table<-c(amplitudes4table,rep(amplitude,4))
                    epochss4table<-c(epochss4table,rep(epochs,4))
                    resolutions4table<-c(resolutions4table,rep(resolution,4))
                    recurrences4table<-c(recurrences4table,rep(recurrence,4))
                    
                    r<-measure_performance_snn(paste0(dataset_name,"/",datasets),nu1,"0.1_ -0.1",n,threshold,decay,amplitude,epochs,resolution,recurrence,fold_ids)
                    
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
            }
          }
        }
      }
    }
  }
}

performance_table<-data.frame(dataset=datasets4table,
                  nu1=nu1s4table,
                  nu2=nu2s4table,
                  n=ns4table,
                  threshold=thresholds4table,
                  decay=decays4table,
                  amplitude=amplitudes4table,
                  epochs=epochss4table,
                  resolution=resolutions4table,
                  recurrent=recurrences4table,
                  average=averages4table,
                  gm=gmeans4table,
                  f1=f1s4table,
                  sensitivity=sensitivities4table,
                  specificity=specificities4table,
                  precision=precisions4table,
                  recall=recalls4table,
                  auc=aucs4table
      )


