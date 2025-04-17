import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from collections import defaultdict
from copy import deepcopy
from scipy.special import softmax
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import time 
from torchmetrics.classification import MulticlassCalibrationError
import torch.nn.functional as F
import torch.nn as nn
from custom_dataset import Dataset_tracked
from aum import AUMCalculator
import pandas as pd

import logging
import math
import numpy as np
import pandas as pd 
import os
import torch
import json 
import random
from multiprocessing import Process, Pool
from torch.multiprocessing import Pool, Process, set_start_method

logger = logging.getLogger('UST')

main_dir = "/home/kgupta27/code/project/"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = "cpu"
print("The device is : ", device)



def multigpu(model):
    model = nn.DataParallel(model).to(device)
    return model 

class BaseModel(torch.nn.Module):
    def __init__(self, checkpoint, num_labels=10):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
        self.T = torch.nn.Parameter(torch.ones(1) * 1.0)


    def forward(self, input_ids, token_type_ids, attention_mask, temperature_scaling=False):
        if temperature_scaling:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            temperature = self.T.unsqueeze(1).expand(outputs.logits.size(0), outputs.logits.size(1))
            outputs.logits /= temperature
        else:
            outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return outputs
    

def evaluate(model, test_dataloader, criterion, batch_size, num_classes, temp_scaling=False):
    full_predictions = []
    true_labels = []
    probabilities = []

    model.eval()
    crt_loss = 0

    with torch.no_grad():
        for elem in tqdm(test_dataloader):
            x = {key: elem[key].to(device)
                for key in elem if key not in ['idx']}
            logits = model(
                input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'], temperature_scaling=temp_scaling)
            results = torch.argmax(logits.logits, dim=1)
            prob = F.softmax(logits.logits.to('cpu'), dim=1)
            probabilities += list(prob)

            crt_loss += criterion(logits.logits, x['lbl']
                                ).cpu().detach().numpy()
            full_predictions = full_predictions + \
                list(results.cpu().detach().numpy())
            true_labels = true_labels + list(elem['lbl'].cpu().detach().numpy())


    model.train()

    metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
    metric = metric.to(device)

    preds = torch.stack(probabilities)
    preds = preds.to(device)

    orig = torch.tensor(true_labels, dtype=torch.float, device=device)

    ece_metric = metric(preds, orig).to(device)

    return f1_score(true_labels, full_predictions, average='macro'), crt_loss / len(test_dataloader), ece_metric


def predict_unlabeled(model, ds_unlabeled):
    model.eval()
    data_loader_unlabeled = torch.utils.data.DataLoader(ds_unlabeled, batch_size=128, shuffle=False) 
    y_pred_unlbl = []  

    with torch.no_grad():
        for elem in data_loader_unlabeled:
                x = {key: elem[key].to(device) for key in elem if key not in ['idx', 'weights']}
                pred = model(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
                y_pred_unlbl.extend(pred.logits.cpu().numpy())

        y_pred_unlbl = np.array(y_pred_unlbl)
        y_pred_unlbl = np.argmax(y_pred_unlbl, axis=-1).flatten()

    pseudolabeled_data = Dataset_tracked(ds_unlabeled.text_list, y_pred_unlbl, ds_unlabeled.idxes, ds_unlabeled.tokenizer, labeled=True)
    return pseudolabeled_data



def train_model_supervised(hist_train, hist_dev, hist_test, pal_test, pt_teacher_checkpoint, cfg,
                    sup_batch_size=32, sup_epochs=10, N_base=10, results_file="results.json", temp_scaling=False, method_type=""):
    logger_dict = {}
    logger_dict["Temperature Scaling"] = temp_scaling

    load_best = False

    model = BaseModel(pt_teacher_checkpoint, num_labels=10)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-2, weight_decay=0.01)
    hist_train_loader = torch.utils.data.DataLoader(hist_train, batch_size=sup_batch_size, shuffle=True)
    hist_dev_loader = torch.utils.data.DataLoader(hist_dev, batch_size=sup_batch_size, shuffle=True)
    hist_test_loader = torch.utils.data.DataLoader(hist_test, batch_size=sup_batch_size, shuffle=True)
    pal_test_loader = torch.utils.data.DataLoader(pal_test, batch_size=sup_batch_size, shuffle=True) 

    cfg.num_labels = 10
    copy_cfg = deepcopy(cfg)
    copy_cfg.attention_probs_dropout_prob = 0.1
    copy_cfg.hidden_dropout_prob = 0.1

    best_f1_overall = 0
    crt_patience = 0

    os.makedirs(method_type, exist_ok=True)

    if load_best == False:
        for counter in range(N_base):
            best_f1 = 0
            copy_cfg.return_dict  = True

            model = BaseModel(pt_teacher_checkpoint)
            model.to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
            if counter == 0:
                logger.info(model)
            for epoch in range(sup_epochs):
                for data in tqdm(hist_train_loader):
                    cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                    optimizer.zero_grad()
                    logits = model(input_ids=cuda_tensors['input_ids'], token_type_ids=cuda_tensors['token_type_ids'], attention_mask=cuda_tensors['attention_mask'])
                    loss = criterion(logits.logits, cuda_tensors['lbl'])
                    loss.backward()
                    optimizer.step()

                f1_macro_validation, loss_validation, ece = evaluate(
                    model, hist_dev_loader, criterion, sup_batch_size, 10)
                
                if f1_macro_validation >= best_f1:
                    crt_patience = 0
                    best_f1 = f1_macro_validation
                    if best_f1 > best_f1_overall:
                        torch.save(model.state_dict(),main_dir + method_type + "/pytorch_model.bin")
                        best_f1_overall = best_f1
                    print('New best macro validation', best_f1, 'Epoch', epoch)
                    continue
            
                if crt_patience == 6:
                    crt_patience = 0
                    print('Exceeding max patience; Exiting..')
                    break

                crt_patience += 1

        del model

    cfg.return_dict = True

    best_model = BaseModel(pt_teacher_checkpoint)
    best_model.to(device)
    state_dict = torch.load(main_dir + method_type + "/pytorch_model.bin")
    best_model.load_state_dict(state_dict)

    f1_macro_test, loss_test, ece_metric = evaluate(best_model, hist_test_loader, criterion, sup_batch_size, 10)
    logger.info ("Historical data macro F1 based on best validation f1 : {}".format(f1_macro_test))

    logger_dict["Best  model"] = {}
    logger_dict["Best  model"]["F1 Historic before temp scaling"] = str(f1_macro_test)
    logger_dict["Best  model"]["ECE Historic before temp scaling"] = str(ece_metric)

    f1_macro_palisades, loss_palisades, ece_metric_palisades = evaluate(best_model, pal_test_loader, criterion, sup_batch_size, 10)
    logger.info ("Palisades macro F1 based on best validation f1 : {}".format(f1_macro_palisades))

    logger_dict["Best  model"]["F1 Palisades before temp scaling"] = str(f1_macro_palisades)
    logger_dict["Best  model"]["ECE Palisades before temp scaling"] = str(ece_metric_palisades)

    logger_dict["Best  model"]["T before temp scaling"] = str(best_model.T.detach().cpu().numpy()[0])

    if temp_scaling:
        optimizer = torch.optim.Adam(best_model.parameters(), lr=2e-02)

        for epoch in range(20):
            for data in tqdm(hist_dev_loader):
                cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                optimizer.zero_grad()
  
                result = best_model(cuda_tensors['input_ids'], cuda_tensors['token_type_ids'], cuda_tensors['attention_mask'], True)

                loss = criterion(result.logits, cuda_tensors['lbl'])
                loss.backward()
                optimizer.step()


    f1_macro_test, loss_test, ece_metric = evaluate(best_model, hist_test_loader, criterion, sup_batch_size, 10, temp_scaling=True)
    logger.info ("Historical data macro F1 based on best validation f1 : {}".format(f1_macro_test))

    logger_dict["Best  model"]["F1 Historic after temp scaling"] = str(f1_macro_test)
    logger_dict["Best  model"]["ECE Historic after temp scaling"] = str(ece_metric)

    f1_macro_palisades, loss_palisades, ece_metric_palisades = evaluate(best_model, pal_test_loader, criterion, sup_batch_size, 10, temp_scaling=True)
    logger.info ("Palisades macro F1 based on best validation f1 : {}".format(f1_macro_palisades))

    logger_dict["Best  model"]["F1 Palisades after temp scaling"] = str(f1_macro_palisades)
    logger_dict["Best  model"]["ECE Palisades after temp scaling"] = str(ece_metric_palisades)

    logger_dict["Best  model"]["T after temp scaling"] = str(best_model.T.detach().cpu().numpy()[0])

    print(json.dumps(logger_dict, indent=4))
    with open(main_dir + method_type+"/"+ results_file + '.txt','w') as fp:
        fp.write(json.dumps(logger_dict, indent=4))



def train_model_self_training(hist_train, hist_dev, hist_test, pal_test, pal_unlabeled, pt_teacher_checkpoint, cfg,
                              unsup_epochs=12,
                    sup_batch_size=32, sup_epochs=10, N_base=10, results_file="results.json", temp_scaling=False, method_type=""):
    logger_dict = {}
    logger_dict["Temperature Scaling"] = temp_scaling

    load_best = False

    model = BaseModel(pt_teacher_checkpoint, num_labels=10)
    model.to(device)
    model.train()

    criterion_supervised = torch.nn.CrossEntropyLoss(reduction='mean')
    criterion_unsupervised = torch.nn.CrossEntropyLoss(reduction='none')

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-2, weight_decay=0.01)
    hist_train_loader = torch.utils.data.DataLoader(hist_train, batch_size=sup_batch_size, shuffle=True)
    hist_dev_loader = torch.utils.data.DataLoader(hist_dev, batch_size=sup_batch_size, shuffle=True)
    hist_test_loader = torch.utils.data.DataLoader(hist_test, batch_size=sup_batch_size, shuffle=True)
    pal_test_loader = torch.utils.data.DataLoader(pal_test, batch_size=sup_batch_size, shuffle=True) 
    pal_unlabeled_loader = torch.utils.data.DataLoader(pal_unlabeled, batch_size=64, shuffle=True)

    cfg.num_labels = 10
    copy_cfg = deepcopy(cfg)
    copy_cfg.attention_probs_dropout_prob = 0.1
    copy_cfg.hidden_dropout_prob = 0.1

    best_f1_overall = 0
    crt_patience = 0

    os.makedirs(method_type, exist_ok=True)

    if load_best == False:
        for counter in range(N_base):
            best_f1 = 0
            copy_cfg.return_dict  = True

            model = BaseModel(pt_teacher_checkpoint)
            model.to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
            if counter == 0:
                logger.info(model)
            for epoch in range(sup_epochs):
                for data in tqdm(hist_train_loader):
                    cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                    optimizer.zero_grad()
                    logits = model(input_ids=cuda_tensors['input_ids'], token_type_ids=cuda_tensors['token_type_ids'], attention_mask=cuda_tensors['attention_mask'])
                    loss = criterion_supervised(logits.logits, cuda_tensors['lbl'])
                    loss.backward()
                    optimizer.step()

                f1_macro_validation, loss_validation, ece = evaluate(
                    model, hist_dev_loader, criterion_supervised, sup_batch_size, 10)
                
                if f1_macro_validation >= best_f1:
                    crt_patience = 0
                    best_f1 = f1_macro_validation
                    if best_f1 > best_f1_overall:
                        torch.save(model.state_dict(),main_dir + method_type + "/pytorch_model.bin")
                        best_f1_overall = best_f1
                    print('New best macro validation', best_f1, 'Epoch', epoch)
                    continue
            
                if crt_patience == 6:
                    crt_patience = 0
                    print('Exceeding max patience; Exiting..')
                    break

                crt_patience += 1

        del model

    cfg.return_dict = True

    # load best teacher model 
    model = BaseModel(pt_teacher_checkpoint)
    model.to(device)
    state_dict = torch.load(main_dir + method_type + "/pytorch_model.bin")
    model.load_state_dict(state_dict)

    for epoch in range(unsup_epochs): #this is just number of self-training steps overall
        pseudolabled_data = predict_unlabeled(model, pal_unlabeled)

        data_sampler = torch.utils.data.RandomSampler(hist_train, num_samples=10**4)
        batch_sampler = torch.utils.data.BatchSampler(data_sampler, sup_batch_size, drop_last=False)
        train_dataloader = torch.utils.data.DataLoader(hist_train, batch_sampler=batch_sampler)
        data_loader_unlabeled = torch.utils.data.DataLoader(pseudolabled_data, batch_size=64, shuffle=False) 

        # resetting these values for each student training iteration
        best_f1_overall = 0
        crt_patience = 0

        for st_epoch in range(sup_epochs): #the student model trains for sup_epochs on the pseudo-labeled data + labeled data
            for data_supervised, data_unsupervised in tqdm(zip(train_dataloader, data_loader_unlabeled)):
                cuda_tensors_supervised = {key: data_supervised[key].to(device) for key in data_supervised if key not in ['idx']}
                
                cuda_tensors_unsupervised = {key: data_unsupervised[key].to(device) for key in data_unsupervised if key not in ['idx']}

                merged_tensors = {}
                for k in cuda_tensors_supervised:
                    merged_tensors[k] = torch.cat((cuda_tensors_supervised[k], cuda_tensors_unsupervised[k]))

                optimizer.zero_grad()

                num_lb = cuda_tensors_supervised['input_ids'].shape[0]

                logits = model(input_ids=merged_tensors['input_ids'], token_type_ids=merged_tensors[
                'token_type_ids'], attention_mask=merged_tensors['attention_mask'])

                logits_lbls = logits.logits[:num_lb]
                logits_ulbl = logits.logits[num_lb:]

                loss_sup = criterion_supervised(logits_lbls, cuda_tensors_supervised['lbl'])
                loss_unsup = criterion_unsupervised(logits_ulbl, cuda_tensors_unsupervised['lbl'])
                loss = 0.5 * loss_sup + 0.5 * torch.mean(loss_unsup)
                loss.backward()
                optimizer.step()

            f1_macro_validation, loss_validation, _ = evaluate(model, hist_dev_loader, criterion_supervised, 
                                                               sup_batch_size, 10)
            if f1_macro_validation >= best_f1:
                crt_patience = 0
                best_f1 = f1_macro_validation

                if best_f1 > best_f1_overall:
                    torch.save(model.state_dict(),main_dir + method_type + "/pytorch_model.bin")
                    best_f1_overall = best_f1
                    print('New best macro validation', best_f1, 'Epoch', epoch)
                    continue

                if crt_patience == 6:
                    crt_patience = 0
                    print('Exceeding max patience; Exiting..')
                    break

                crt_patience += 1


    # load best student model after all the self-training steps 
    copy_cfg.return_dict = True
    best_model = BaseModel(pt_teacher_checkpoint)
    best_model.to(device)
    state_dict = torch.load(main_dir + method_type + "/pytorch_model.bin")
    best_model.load_state_dict(state_dict)



    f1_macro_test, loss_test, ece_metric = evaluate(best_model, hist_test_loader, criterion_supervised, sup_batch_size, 10)
    logger.info ("Historical data macro F1 based on best validation f1 : {}".format(f1_macro_test))

    logger_dict["Best  model"] = {}
    logger_dict["Best  model"]["F1 Historic before temp scaling"] = str(f1_macro_test)
    logger_dict["Best  model"]["ECE Historic before temp scaling"] = str(ece_metric)

    f1_macro_palisades, loss_palisades, ece_metric_palisades = evaluate(best_model, pal_test_loader, criterion_supervised, sup_batch_size, 10)
    logger.info ("Palisades macro F1 based on best validation f1 : {}".format(f1_macro_palisades))

    logger_dict["Best  model"]["F1 Palisades before temp scaling"] = str(f1_macro_palisades)
    logger_dict["Best  model"]["ECE Palisades before temp scaling"] = str(ece_metric_palisades)

    logger_dict["Best  model"]["T before temp scaling"] = str(best_model.T.detach().cpu().numpy()[0])

    if temp_scaling:
        optimizer = torch.optim.Adam(best_model.parameters(), lr=2e-02)

        for epoch in range(20):
            for data in tqdm(hist_dev_loader):
                cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                optimizer.zero_grad()
  
                result = best_model(cuda_tensors['input_ids'], cuda_tensors['token_type_ids'], cuda_tensors['attention_mask'], True)

                loss = criterion_supervised(result.logits, cuda_tensors['lbl'])
                loss.backward()
                optimizer.step()


    f1_macro_test, loss_test, ece_metric = evaluate(best_model, hist_test_loader, criterion_supervised, sup_batch_size, 10, temp_scaling=True)
    logger.info ("Historical data macro F1 based on best validation f1 : {}".format(f1_macro_test))

    logger_dict["Best  model"]["F1 Historic after temp scaling"] = str(f1_macro_test)
    logger_dict["Best  model"]["ECE Historic after temp scaling"] = str(ece_metric)

    f1_macro_palisades, loss_palisades, ece_metric_palisades = evaluate(best_model, pal_test_loader, criterion_supervised, sup_batch_size, 10, temp_scaling=True)
    logger.info ("Palisades macro F1 based on best validation f1 : {}".format(f1_macro_palisades))

    logger_dict["Best  model"]["F1 Palisades after temp scaling"] = str(f1_macro_palisades)
    logger_dict["Best  model"]["ECE Palisades after temp scaling"] = str(ece_metric_palisades)

    logger_dict["Best  model"]["T after temp scaling"] = str(best_model.T.detach().cpu().numpy()[0])

    print(json.dumps(logger_dict, indent=4))
    with open(main_dir + method_type+"/"+ results_file + '.txt','w') as fp:
        fp.write(json.dumps(logger_dict, indent=4))

