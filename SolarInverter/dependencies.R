######################################################
#                                                    #
#-----------Functions to help in evaluation----------#
#                                                    #
######################################################

#Libraries
library(pROC)
library(caret)
library(pracma)

#Function to compute AUC
get_auc<-function(label,pred){
  auc(label,pred,direction="<",levels=c("0","1"))
}

#Function to compute F1-score and GM
get_metric<-function(label,pred,metric){
  
  #Define thresholds:
  ths<-seq(min(pred),max(pred),(max(pred)-min(pred))/10)
  metric_tmp<-c()
  for (th in ths){
    #Discretizes preds and labels according to each threshold:
    pred_tmp<-pred
    pred_tmp[pred_tmp<=th]<-0
    pred_tmp[pred_tmp>0]<-1
    #Transforms into factors:
    pred_tmp<-factor(pred_tmp,levels=c(0,1))
    label_tmp<-factor(label,levels=c(0,1))
    #Generates confussion matrix:
    matr<-confusionMatrix(pred_tmp,label_tmp,positive="1")
    #Computes the metric:
    if (metric=="gm"){
      metric_tmp<-c(metric_tmp,sqrt(matr$byClass["Sensitivity"]*matr$byClass["Specificity"]))
    }else if (metric=="f1"){
      value_f1<-matr$byClass["F1"]
      if (is.na(value_f1)){
        metric_tmp<-c(metric_tmp,0)
      }else{
        metric_tmp<-c(metric_tmp,value_f1)
      }
    }
  }
  #Returns the maximum value.
  max(metric_tmp,na.rm=TRUE)
}

#Function to preprocess data (average spikes, etc.).
preprocess_data<-function(spikes,label,average){
  if (average>1){
    spikes<-movavg(spikes,average)
  }
  label<-label[1:length(spikes)]
  #Return preprocessed spikes and labels.
  list(label=label,pred=spikes)
}
