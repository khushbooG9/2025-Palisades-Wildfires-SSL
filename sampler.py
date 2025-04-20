"""
Author: Subhabrata Mukherjee (submukhe@microsoft.com)
Code for Uncertainty-aware Self-training (UST) for few-shot learning.
Adapted from https://github.com/microsoft/UST
"""

from sklearn.utils import shuffle
from custom_dataset import Dataset_tracked

import logging
import numpy as np
import os
import random


logger = logging.getLogger('UST')

def get_BALD_acquisition(y_T):

	expected_entropy = - np.mean(np.sum(y_T * np.log(y_T + 1e-10), axis=-1), axis=0) 
	expected_p = np.mean(y_T, axis=0)
	entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)
	return (entropy_expected_p - expected_entropy)

def sample_by_bald_difficulty(tokenizer, X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by difficulty BALD acquisition function")
	BALD_acq = get_BALD_acquisition(y_T)
	p_norm = np.maximum(np.zeros(len(BALD_acq)), BALD_acq)
	p_norm = p_norm / np.sum(p_norm)
	indices = np.random.choice(len(X['input_ids']), num_samples, p=p_norm, replace=False)
	X_s = {"input_ids": X["input_ids"][indices], "token_type_ids": X["token_type_ids"][indices], "attention_mask": X["attention_mask"][indices]}
	y_s = y[indices]
	w_s = y_var[indices][:,0]
	return X_s, y_s, w_s


def sample_by_bald_easiness(tokenizer, X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by easy BALD acquisition function")
	BALD_acq = get_BALD_acquisition(y_T)
	p_norm = np.maximum(np.zeros(len(BALD_acq)), (1. - BALD_acq)/np.sum(1. - BALD_acq))
	p_norm = p_norm / np.sum(p_norm)
	logger.info (p_norm[:10])
	indices = np.random.choice(len(X['input_ids']), num_samples, p=p_norm, replace=False)
	X_s = {"input_ids": X["input_ids"][indices], "token_type_ids": X["token_type_ids"][indices], "attention_mask": X["attention_mask"][indices]}
	y_s = y[indices]
	w_s = y_var[indices][:,0]
	return X_s, y_s, w_s


def sample_by_bald_class_easiness(X_new_unlabeled_dataset, y_mean, y_var, y_pred, unsup_size, num_classes, y_T):
	
	logger.info ("Sampling by easy BALD acquisition function per class")

	BALD_acq = get_BALD_acquisition(y_T)
	BALD_acq = (1. - BALD_acq)/np.sum(1. - BALD_acq)
	logger.info (BALD_acq)
	samples_per_class = unsup_size // num_classes


	indices = []
	y_s = []
	w_s = []
	x_s = []
	i_s = []
	for label in range(num_classes):
		y_ = y_pred[y_pred==label]
		y_var_ = y_var[y_pred == label]
		p_norm = BALD_acq[y_pred==label]
		p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
		p_norm = p_norm/np.sum(p_norm)
		pool_of_idx = np.array(X_new_unlabeled_dataset.idxes)[y_pred == label]
		map_pool_to_idx = dict()
		map_idx_to_pool = dict()
		for p, i in zip(pool_of_idx, range(len(pool_of_idx))):
			map_pool_to_idx[p] = i
			map_idx_to_pool[i] = p
		print("MAP ", map_pool_to_idx)
		if len(pool_of_idx) ==0:
			continue
		if len(pool_of_idx) < samples_per_class:
			logger.info ("Sampling with replacement.")
			replace = True
		else:
			replace = False
		print("Pool of idx ", len(pool_of_idx))
		indices = np.random.choice(pool_of_idx, samples_per_class, p=p_norm, replace=replace)
		# print("Indices ", indices)
		# mapper = {}
		# crt = 0
		# for idx in indices:
		# 	mapper[idx] = crt
		# 	crt += 1

		pool_dataset = X_new_unlabeled_dataset.get_subset_dataset(pool_of_idx)
		x_s.extend([pool_dataset.text_list[map_pool_to_idx[i]] for i in indices])
		i_s.extend([pool_dataset.idxes[map_pool_to_idx[i]] for i in indices])

		indices_int = [int(i) for i in indices]

		y_s.extend([label] * len(indices_int))
		#y_s.extend(y_pred[indices_int])
		for i in indices:
			w_s.append(y_var_[map_pool_to_idx[i], label])
		# w_s.extend([y_var_[map_pool_to_idx[i]][label] for i in indices ])
		# x_s.extend([pool_dataset.text_list[map_pool_to_idx[i]] for i in indices])
		# i_s.extend([pool_dataset.idxes[map_pool_to_idx[i]] for i in indices])
		assert len(y_s) == len(w_s) and len(w_s) == len(x_s) and len(x_s) == len(i_s)

	text_lists, labels, weights, ids = shuffle(x_s, y_s, w_s, i_s)
	final_ds = Dataset_tracked(text_lists, labels, ids, X_new_unlabeled_dataset.tokenizer, labeled=True)
	final_ds.weights = weights

	return final_ds


def sample_by_bald_class_difficulty(tokenizer, X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by difficulty BALD acquisition function per class")
	BALD_acq = get_BALD_acquisition(y_T)
	samples_per_class = num_samples // num_classes
	X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = [], [], [], [], []
	for label in range(num_classes):
		X_input_ids, X_token_type_ids, X_attention_mask = X['input_ids'][y == label], X['token_type_ids'][y == label], X['attention_mask'][y == label]
		y_ = y[y==label]
		y_var_ = y_var[y == label]		
		p_norm = BALD_acq[y==label]
		p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
		p_norm = p_norm/np.sum(p_norm)
		if len(X_input_ids) < samples_per_class:
			replace = True
			logger.info ("Sampling with replacement.")
		else:
			replace = False
		indices = np.random.choice(len(X_input_ids), samples_per_class, p=p_norm, replace=replace)
		X_s_input_ids.extend(X_input_ids[indices])
		X_s_token_type_ids.extend(X_token_type_ids[indices])
		X_s_attention_mask.extend(X_attention_mask[indices])
		y_s.extend(y_[indices])
		w_s.extend(y_var_[indices][:,0])
	X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s)
	return {'input_ids': np.array(X_s_input_ids), 'token_type_ids': np.array(X_s_token_type_ids), 'attention_mask': np.array(X_s_attention_mask)}, np.array(y_s), np.array(w_s)
