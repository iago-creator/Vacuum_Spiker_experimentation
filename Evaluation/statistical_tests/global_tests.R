#Code to apply the Iman-Davenport and Wilcoxon signed-rank test to the obtained results.

#Load libraries
library(scmamp)


apply_wilcoxon<-function(data){

  #Function to apply Wilcoxon test, compared to the Vacuum Spiker results, saved in the first column.
  p_values<-c()
  
  for(i in 2:ncol(data)){
    p_values<-c(p_values,wilcox.test(data[,"snn"],data[,i],paired = TRUE,correct=FALSE)$p.value)
  }
  
  p_values
}

#Load results from all datasets
data<-data.frame()

for(dataset_origin in main_datasets){

  data_tmp<-read.csv(paste0("results/",dataset_origin,"/",metric,"/comparison.csv"))
  
  data<-rbind(data,data_tmp)
}

#Suppress first column (dataset name)
data<-data[,-1]

#Separate performance and MACs
data_r<-as.matrix(data[,gsub("macs_","",names(data),fixed=TRUE)==names(data)])
data_e<-as.matrix(data[,!gsub("macs_","",names(data),fixed=TRUE)==names(data)])

#Adapt data to the same format (column names, erasing NAs and infinites, etc.)
colnames(data_e)<-gsub("macs_","",colnames(data_e),fixed=TRUE)
data_e[data_r==(-1) | is.infinite(data_r)]<-NA
data_r[data_r==(-1) | is.infinite(data_r)]<-NA

#Suppress experiments with troubles
data_r<-data_r[!is.na(rowSums(data_r)),]
data_e<-data_e[!is.na(rowSums(data_e)),]

print(paste0("We have ",nrow(data_r)," experiments for the metric ",metric))

#Check global differences with ImanDavenport test
iman_r <- imanDavenportTest(data_r)
iman_e <- imanDavenportTest(data_e)

#Print results
cat("Iman-Davenport test p-value for performance: ", format(iman_r$p.value,scientific=TRUE), "\n")

cat("Iman-Davenport test p-value for energy consumption: ",format(iman_e$p.value,scientific=TRUE) , "\n")

#Apply Wilcoxon test if the global differences are significant:
if (iman_r$p.value < 0.05) {
  cat("There are global significant differences in performance. Applying post-hoc...\n")
  
  #Apply Wilcoxon test
  p_values<-apply_wilcoxon(data_r)
  #Holm correction
  adjusted_pvals <- c(-1,p.adjust(p_values, method = "holm"))
  
  names(adjusted_pvals)<-colnames(data_r)
  print(paste0("Adjusted p-vals for performance in ",metric))
  print(adjusted_pvals)
  #Compute medians for performance
  medians<-c()
  for (i in 1:ncol(data_r)){medians<-c(medians,median(data_r[,i]))}
  
  names(medians)<-colnames(data_r)
  print(paste0("Medians for performance in ",metric))
  print(medians)
  
} else {
  cat("There are no global significant differences in performance.\n")
}

if (iman_e$p.value < 0.05) {
  cat("There are global significant differences in consumption. Applying post-hoc...\n")
  
  # Post-hoc con test de Wilcoxon, usando rangos, y corrección de Holm
  p_values<-apply_wilcoxon(data_e)
  
  adjusted_pvals <- c(-1,p.adjust(p_values, method = "holm"))
  
  names(adjusted_pvals)<-colnames(data_e)
  print(paste0("Adjusted p-vals for MACs when ",metric," is used to select models"))
  print(adjusted_pvals)
  
  medians<-c()
  for (i in 1:ncol(data_e)){medians<-c(medians,median(data_e[,i]))}
  
  #Comvert MACs to thousands of MACs
  medians<-medians/1000
  
  names(medians)<-colnames(data_r)
  print(paste0("Medians for MACs when ",metric," is used to select models"))
  print(medians)
} else {
  cat("There are no global significant differences in consumption.\n")
}
