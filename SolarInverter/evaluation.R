######################################################
#                                                    #
#----Code to run evaluation of obtained results------#
#                                                    #
######################################################

#Source dependencies
source("dependencies.R")

#Set the configuration to eval
configuration<-"_exc_inh"

#Load spikes
spikes<-as.vector(read.csv(paste0("outputs/spikes_",configuration),header=FALSE)$V1)

#Load labels
test<-read.csv("data/test.csv")

label<-test$Label

#Compute performance metrics
auc<-0
f1<-0
gm<-0
for(avg in c(1,100,200,300)){
  data<-preprocess_data(spikes,label,avg)
  auc_tmp<-get_auc(data$label,data$pred)
  f1_tmp<-get_metrica(data$label,data$pred,"f1")
  gm_tmp<-get_metrica(data$label,data$pred,"gm")
  if(auc_tmp>auc){auc<-auc_tmp}
  if(f1_tmp>f1){f1<-f1_tmp}
  if(gm_tmp>gm){gm<-gm_tmp}
}

print("The following results have been obtained:")
print(paste0("AUC: ",auc_ppal))
print(paste0("F1: ",f1_ppal))
print(paste0("GM: ",gm_ppal))

#Compute energy metrics.
print(paste0("Mean spikes produced: ",mean(spikes)))
