#!/bin/bash


# experiments for the supervised method, 3 different seeds with each picking best model out of 3 runs
# CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=40 python3 main.py  --aum_save_dir AUM1 --results_file sup_1 --method_type supervised
# CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=41 python3 main.py  --aum_save_dir AUM1 --results_file sup_2 --method_type supervised
# CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=42 python3 main.py  --aum_save_dir AUM1 --results_file sup_3 --method_type supervised

# experiments for the self-training method, 3 different seeds with each picking base model out of 3 runs
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=40 python3 main.py  --aum_save_dir AUM1 --results_file ST_1 --method_type self_training
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=41 python3 main.py  --aum_save_dir AUM1 --results_file ST_2 --method_type self_training
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=42 python3 main.py  --aum_save_dir AUM1 --results_file ST_3 --method_type self_training
