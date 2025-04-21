# 2025-Palisades-Wildfires-SSL
-------------------------------------------
## Important file details - 
* clean_palisades.py - all the code containing to get the final usable data files (more details in data folder). Lot of code is commented but can be reused by uncommenting if needed.
* run_exp.sh - it contains python commands for all the experiments. Can be updated easily to include more experiments with various hyperparameters. 
* custom_dataset.py - custom pytorch dataset class for properly using the datasets for method implementation. 
* main.py - code to put everything together for entry point for running experiments. It sets up all the arguments needed. 
* train.py - contains individual functions for all methods and related reusable functions. 
* sampler.py - sampling code for UST, adapted from https://github.com/microsoft/UST
-------------------------------------------
## How to run the experiments - 
* A shell script (run_exp.sh) for running all the experiments related to each method is provided. Please just uncomment the set of experiments that need to be run. (Assuming a CUDA Pytorch enviroment all important nlp related packages is already setup.)
* The shell script includes default parameters, but you can easily modify various hyperparameters. Check out main.py for the possible arguments and relevant details. 
-------------------------------------------
## Result files - 
* Each method such as supervised, UST, mixmatch etc has it's own folder for results.
* Every method has 3 random experiments with a starting teacher model is selected from N_base=3.
* Each result file is a text file and have data for both F1 and ECE on historical data and palisades data, before and after temperature scaling (explicit model calibration method)
* I report the average of 3 runs for each method. 
-------------------------------------------
### *Note - Every folder has it's own readme with details needed. 
