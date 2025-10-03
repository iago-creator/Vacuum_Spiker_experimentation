#Code to apply Chi-Squared test to results, in order to study the synaptic behaviour.

for(metric in c("auc","gm","f1")){
  
  #Read parameters for the best found configurations
  a1<-read.csv(paste0("results/Numenta/",metric,"/parameters/snn.csv"))
  a2<-read.csv(paste0("results/CalIt2/",metric,"/parameters/snn.csv"))
  a3<-read.csv(paste0("results/Dodgers/",metric,"/parameters/snn.csv"))
  
  #Create a single dataframe
  a<-rbind(a1,a2)
  a<-rbind(a,a3)
  
  #Skip missing values
  a<-a[!is.na(a[[metric]]) & !is.infinite(a[[metric]]) & !(a[[metric]]==-1),]
  
  #Label the absence of recurrent connection
  a$nu2[is.na(a$nu2)]<-"not"
  
  #Label the different configurations as potentiation (exc), depresion (inh) and balanced (neu)
  a$nu1[(a$nu1=="-0.1_ 0.1") | (a$nu1=="-0.1_ -0.1")]<-"exc"
  a$nu1[(a$nu1=="0.1_ -0.1") | (a$nu1=="0.1_ 0.1")]<-"inh"
  a$nu2[(a$nu2=="-0.1_ 0.1")]<-"exc"
  a$nu2[(a$nu2=="0.1_ -0.1")]<-"inh"
  a$nu2[(a$nu2=="0.1_ 0.1") | (a$nu2=="-0.1_ -0.1")]<-"neu"
  
  #Create a single label for each configuration
  mu<-paste(a$nu1,a$nu2,sep=" ")
  
  #Compute frequencies
  tbl <- table(mu)
  
  #Create a single set with the lowest frequent synaptic behaviours
  mu_mod <- ifelse(tbl[as.character(mu)] < 5, "Other", as.character(mu))
  tbl_mod <- table(mu_mod)
  
  #Apply Chi-Squared test
  test<-chisq.test(tbl_mod)
  print(paste0("P-value for ",metric,": ",test$p.value))
  
  #Create the table with residuals
  table_chi<-rbind(tbl_mod,test$expected,test$residuals,test$stdres)
  rownames(table_chi)<-c("freqs","expected","Pearson residuals","Standardized residuals")
  
  print(table_chi)
}

