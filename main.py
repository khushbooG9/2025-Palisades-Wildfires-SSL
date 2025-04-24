from distutils.command.config import config
from sklearn.utils import shuffle

import argparse
import logging
import numpy as np
import os
import pandas as pd
import random
import sys
import transformers

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from custom_dataset import Dataset_tracked
from train import train_model_supervised, train_model_self_training, train_model_mixmatch, train_model_ust, train_model_aumst
from train_moex import train_model_aumst_moex_mixup


# logging
logger = logging.getLogger('PALISADES')
logging.basicConfig(level = logging.INFO)

GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
logger.info ("Global seed {}".format(GLOBAL_SEED))


label_to_id = {
	"caution_and_advice":0,
	"displaced_people_and_evacuations":1,
	"infrastructure_and_utility_damage":2,
	"injured_or_dead_people":3,
	"missing_or_found_people":4,
	"not_humanitarian":5,
	"other_relevant_information":6,
	"requests_or_urgent_needs":7,
	"rescue_volunteering_or_donation_effort":8,
	"sympathy_and_support":9, 
}

def get_dataset(path, tokenizer, labeled=True):

    df = pd.read_csv(path, sep='\t')
    text_list = []
    labels_list = []
    ids_list = []
    for i, row in df.iterrows():
        if pd.isna(row['tweet_text']):
            continue
        text_list.append(row['tweet_text'])
        if labeled:
            labels_list.append(label_to_id[row['class_label']])
        else:
            labels_list.append(-1)
        ids_list.append(row['tweet_id'])
        
    dataset = Dataset_tracked(text_list, labels_list, ids_list, tokenizer, labeled=labeled)       
    return dataset

if __name__ == '__main__':

	# construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--aum_save_dir", nargs="?", type=str, default="AUM0", help="Aum save directory")
    parser.add_argument("--method_type", nargs="?", type=str, default="supervised", help="type of training approach")
    parser.add_argument("--seq_len", nargs="?", type=int, default=128, help="sequence length")
    parser.add_argument("--sup_batch_size", nargs="?", type=int, default=20, help="batch size for fine-tuning base model")
    parser.add_argument("--unsup_batch_size", nargs="?", type=int, default=64, help="batch size for self-training on pseudo-labeled data")
    parser.add_argument("--sample_size", nargs="?", type=int, default=1800, help="number of unlabeled samples for evaluating uncetainty on in each self-training iteration")
    parser.add_argument("--unsup_size", nargs="?", type=int, default=1000, help="number of pseudo-labeled instances drawn from sample_size and used in each self-training iteration")
    parser.add_argument("--sample_scheme", nargs="?", default="easy_bald_class_conf", help="Sampling scheme to use")
    parser.add_argument("--T", nargs="?", type=int, default=30, help="number of masked models for uncertainty estimation")
    parser.add_argument("--alpha", nargs="?", type=float, default=0.1, help="hyper-parameter for confident training loss")
    parser.add_argument("--sup_epochs", nargs="?", type=int, default=18, help="number of epochs for fine-tuning base model")
    parser.add_argument("--unsup_epochs", nargs="?", type=int, default=12, help="number of self-training iterations")
    parser.add_argument("--N_base", nargs="?", type=int, default=3, help="number of times to randomly initialize and fine-tune few-shot base encoder to select the best starting configuration")
    parser.add_argument("--pt_teacher_checkpoint", nargs="?", default="vinai/bertweet-base", help="teacher model checkpoint to load pre-trained weights")
    parser.add_argument("--results_file", nargs="?", default="result.txt", help="file name")
    parser.add_argument("--hidden_dropout_prob", nargs="?", type=float, default=0.3, help="dropout probability for hidden layer of teacher model")
    parser.add_argument("--attention_probs_dropout_prob", nargs="?", type=float, default=0.3, help="dropout probability for attention layer of teacher model")
    parser.add_argument("--dense_dropout", nargs="?", type=float, default=0.5, help="dropout probability for final layers of teacher model")
    parser.add_argument("--temp_scaling", nargs="?", type=bool, default=True, help="temp scaling" )
    parser.add_argument("--saliency", nargs="?", type=bool, default=True, help="saliency")

    args = vars(parser.parse_args())
    logger.info(args)

    max_seq_length = args["seq_len"]
    sup_batch_size = args["sup_batch_size"]
    unsup_batch_size = args["unsup_batch_size"]
    unsup_size = args["unsup_size"]
    sample_size = args["sample_size"]
    aum_save_dir = args["aum_save_dir"]
    sample_scheme = args["sample_scheme"]
    T = args["T"]
    alpha = args["alpha"]
    sup_epochs = args["sup_epochs"]
    unsup_epochs = args["unsup_epochs"]
    N_base = args["N_base"]
    pt_teacher_checkpoint = args["pt_teacher_checkpoint"]
    dense_dropout = args["dense_dropout"]
    attention_probs_dropout_prob = args["attention_probs_dropout_prob"]
    hidden_dropout_prob = args["hidden_dropout_prob"]
    results_file_name = args["results_file"]
    temp_scaling = args["temp_scaling"]


    cfg = AutoConfig.from_pretrained(pt_teacher_checkpoint)
    cfg.hidden_dropout_prob = hidden_dropout_prob
    cfg.attention_probs_dropout_prob = attention_probs_dropout_prob

    tokenizer = AutoTokenizer.from_pretrained(pt_teacher_checkpoint)

    main_dir = "/home/kgupta27/code/project/"

    history_train = get_dataset(main_dir + "data/historical_train.tsv", tokenizer)
    history_dev = get_dataset(main_dir + "data/historical_dev.tsv", tokenizer)
    history_test = get_dataset(main_dir + "data/historical_test.tsv", tokenizer)
    palisades_test = get_dataset(main_dir + "data/palisades_test.tsv", tokenizer)
    palisades_unlabeled = get_dataset(main_dir + "data/palisades_unlabeled.tsv", tokenizer, labeled=False)

    if args["method_type"] == "supervised":
        train_model_supervised(history_train, history_dev, history_test, palisades_test, pt_teacher_checkpoint, cfg, 
                    sup_batch_size=sup_batch_size, sup_epochs=sup_epochs, N_base=N_base, results_file=results_file_name, 
                    temp_scaling=temp_scaling, method_type=args["method_type"]) 
        
    elif args["method_type"] == "self_training":
        train_model_self_training(history_train, history_dev, history_test, palisades_test, palisades_unlabeled,pt_teacher_checkpoint, 
                                  cfg, unsup_epochs=unsup_epochs, sup_batch_size=sup_batch_size, sup_epochs=sup_epochs, 
                                  N_base=N_base, results_file=results_file_name, temp_scaling=temp_scaling, method_type=args["method_type"])

    elif args["method_type"] == "mixmatch":
         train_model_mixmatch(history_train, history_dev, history_test, palisades_test, palisades_unlabeled,pt_teacher_checkpoint, 
                                  cfg, unsup_epochs=unsup_epochs, sup_batch_size=sup_batch_size, sup_epochs=sup_epochs, 
                                  N_base=N_base, results_file=results_file_name, temp_scaling=temp_scaling, method_type=args["method_type"])
         
    elif args["method_type"] == "ust":
        train_model_ust(history_train, history_dev, history_test, palisades_test, palisades_unlabeled,pt_teacher_checkpoint, 
                                  cfg, unsup_epochs=unsup_epochs, T=args["T"], alpha=0.1, dense_dropout=0.5, 
                                  sup_batch_size=sup_batch_size, sup_epochs=sup_epochs, 
                                  N_base=N_base, results_file=results_file_name, temp_scaling=temp_scaling, method_type=args["method_type"])
        
    elif args["method_type"] == "aum_st":
        train_model_aumst(history_train, history_dev, history_test, palisades_test, palisades_unlabeled,pt_teacher_checkpoint, 
                                  cfg, unsup_epochs=unsup_epochs, sup_batch_size=sup_batch_size, sup_epochs=sup_epochs, 
                                  N_base=N_base, results_file=results_file_name, temp_scaling=temp_scaling, method_type=args["method_type"])
    
    elif args["method_type"] == "moex_mixup":
        train_model_aumst_moex_mixup(history_train, history_dev, history_test, palisades_test, palisades_unlabeled,pt_teacher_checkpoint, 
                                  cfg, unsup_epochs=unsup_epochs, lam_moex=0.5, sup_batch_size=sup_batch_size, sup_epochs=sup_epochs, 
                                  N_base=N_base, results_file=results_file_name, temp_scaling=temp_scaling, method_type=args["method_type"])
        
    else:
        print("Invalid method type. Please choose from supervised, self_training, mixmatch, ust, aum_st, moex_mixup")
        sys.exit(1)

