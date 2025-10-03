## IAGO — Time Series Anomaly Analysis with SNN and Baselines

Project for anomaly detection in time series comparing the proposed Vacuum Spiker algorithm, based on Spiking Neural Networks (SNNs), against multiple Deep Learning and classic Machine Learning baselines. It includes a reproducible evaluation pipeline (R) for performance and approximate computational cost, through MACs counting.

### Main contents
- Proposed algorithm: 'Vacuum_Spiker/'
- Deep Learning baselines: 'Baselines/DeepLearning_baselines/'
- Machine Learning baselines: 'Baselines/MachineLearning_baselines/'
- Evaluation and tables: 'Evaluation/' (R scripts, metric/table generation)
- Input data: 'input/' with three sources ('CalIt2', 'Dodgers', 'Numenta')

## Repository structure

- '/'
  - 'Vacuum_Spiker/'
    - 'execute_experimentation.py' (K-fold training/validation loop for SNN)
    - 'dependencies.py' (SNN creation in BindsNET, encoding, utilities)
    - 'ppal.py' (experiment grid launcher for SNN)
  - 'Baselines/'
    - 'DeepLearning_baselines/'
      - 'execute_experimentation.py' (K-fold training/validation for DL)
      - 'baselines.py' (adapted models: OhShuLih, CaiWenjuan, ZhengZhenyu, YildirimOzal, LSTMAutoencoder, Conv1dAutoencoder)
      - 'TSFEDLtorch/' (auxiliary PyTorch blocks/utilities)
      - 'ppal.py' (grid launcher for DL)
    - 'MachineLearning_baselines/'
      - 'exp_lof.py', 'ppal_lof.py' (Local Outlier Factor)
      - 'exp_ocsvm.py', 'ppal_ocsvm.py' (One Class SVM)
  - 'Evaluation/'
    - 'ppal.R' (computation of performance and energy metrics, search of best results, table generation, statistical tests)
    - 'dependencies/' (R: energy/performance measurements)
    - 'search_codes/' (R: family-specific aggregation)
    - 'statistical_tests/' (R: comparisons and tests)
    - 'tables/' (CSV outputs)
  - 'input/' (source-labelled CSVs)

## Input data

This project uses three data sources in 'input/', each with labelled CSVs ('value', 'label'):

- 'CalIt2/' — people counts. Citation: Hutchins, J. (2006). CalIt2 Building People Counts [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NG78.
- 'Dodgers/' — traffic loop sensors. Citation: Hutchins, J. (2006). Dodgers Loop Sensor [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51P50.
- 'Numenta/' — multiple labelled series (NAB). See CSVs in the subfolder. Citation: Related reference on real-time anomaly detection: Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. Neurocomputing.

The input datasets have been preprocessed to ensure that the separation between successive instances is homogeneous, trying to keep as many samples as possible.

## Evaluation and results ('Evaluation/')

- Main orchestrator: 'ppal.R'
  - Compiles energy and performance measurement routines: 'dependencies/energy_measurement.R', 'dependencies/performance_measurement.R'.
  - Aggregates results by family:
    - SNN: 'search_codes/performance_search_snn.R', 'search_codes/spikes_search_snn.R' + MACs estimation 'estimate_snn_macs'.
    - Deep Learning: 'search_codes/performance_search_ann.R' + 'estimate_ann_macs'.
    - Classic ML: 'search_codes/performance_search_ml.R' + 'estimate_ml_macs'.
  - Writes per-source tables to 'tables/' ('*_snn.csv', '*_ann.csv', '*_ml.csv').
  - Selects best results per metric ('gm', 'auc', 'f1') with 'search_codes/search_best_results.R', and get the corresponding energy consumption.
  - Global statistical tests: 'statistical_tests/global_tests.R', to study differences in performance and energy consumption between Vacuum Spiker algorithm and baselines; and 'synanptic_behaviour_study.R', to study the different synaptic behaviours.

## Requirements

- Python 3.8+ with:
  - 'torch', 'numpy', 'pandas', 'scikit-learn', 'bindsnet'

The library versions used in the execution of this project have been the following:
  - 'torch 2.4.0', 'numpy 2.0.2', 'pandas 2.2.2', 'scikit-learn 1.5.2', 'bindsnet 0.2.7'
- R 4.1+ with the following libraries:
dplyr,purrr,scmamp,pracma,pROC,caret

The versions used during the execution of the present project have been the following:
'dplyr 1.1.4','purrr 1.0.4','scmamp 0.3.2','pracma 2.4.4','pROC 1.18.5','caret 7.0.1'

## How to run (summary)

1) Copy the folder 'input' into the Vacuum Spiker and baselines folders.

2) Run experiment grids:

'''bash
# SNN (Vacuum Spiker)
cd /Vacuum_Spiker
python ppal.py

# Deep Learning
cd ../Baselines/DeepLearning_baselines
python ppal.py

# Machine Learning: LOF and OCSVM
cd ../MachineLearning_baselines
python ppal_lof.py
python ppal_ocsvm.py
'''
The script 'count_macs.py' in DeepLearningBaselines can be run by itself. It outputs a code that can be copied in the 'Evaluation/dependencies/energy_measurement.R' file, to include the MACs estimates for DL baselines in the tables and the statistical tests.

3) Generate tables and analysis in R:

Place the outputs from DLBaselines, MLBaselines and VacuumSpiker in the folder /outputs, inside the Evaluation folder.
'''r
setwd("/Evaluation")
source("ppal.R")
'''

The resulting tables are stored in '/Evaluation/tables/'. The results of the statistical tests are displayed on the screen.

## Notes and conventions

- Output paths include hyperparameters to enable automatic aggregation.

## References

- Hutchins, J. (2006). CalIt2 Building People Counts [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NG78.
- Hutchins, J. (2006). Dodgers Loop Sensor [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51P50.
- Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. Neurocomputing.
