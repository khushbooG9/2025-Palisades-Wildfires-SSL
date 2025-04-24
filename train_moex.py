
import pandas as pd
import json
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from aum import AUMCalculator
from multiprocessing import Process, Pool
from torch.multiprocessing import Pool, Process, set_start_method
from torchmetrics.classification import MulticlassCalibrationError
import os
import logging
from custom_dataset import Dataset_tracked

logger = logging.getLogger('521 Experiments')

main_dir = "/home/kgupta27/code/project/"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = "cpu"
print("The device is : ", device)

class PONO(nn.Module):
    def __init__(self, input_size=None, return_stats=False, affine=True, eps=1e-5):
        super(PONO, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std
    

class MomentShortcut(nn.Module):
    def __init__(self, beta=None, gamma=None):
        super(MomentShortcut, self).__init__()
        self.gamma, self.beta = gamma, beta

    def forward(self, x, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x
    


class MoexModel(nn.Module):
    """Pre-trained model for classification."""

    def __init__(self, checkpoint, num_labels=10, pono=True, ms=True):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
        self.classifier = nn.Linear(768, num_labels)
        self.pono = PONO(affine=False) if pono else None
        self.ms = MomentShortcut() if ms else None
        self.T = torch.nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, input_ids, token_type_ids, attention_mask, input_ids2=None
                , token_type_ids2=None, attention_mask2=None, temp_scaling=False):
        
        if temp_scaling:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            temperature = self.T.unsqueeze(1).expand(outputs.logits.size(0), outputs.logits.size(1))
            outputs.logits /= temperature
        else:
            outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        
        cls_output = outputs.logits #outputs[0][:, 0]
        #print("Output before moex : ", cls_output)

        # apply moex here 
        if input_ids2 is not None:
            if temp_scaling:
                with torch.no_grad():
                    outputs2 = self.model(input_ids=input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)
                temperature2 = self.T.unsqueeze(1).expand(outputs2.logits.size(0), outputs2.logits.size(1))
                outputs2.logits /= temperature2
            else:
                outputs2 = self.model(input_ids=input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)

            cls_output2 = outputs2.logits #outputs2[0][:,0]

            cls_output, _, _ = self.pono(cls_output)
            #print("Output during moex : ", cls_output)
            cls_output2, mean, std = self.pono(cls_output2)
            cls_output = self.ms(cls_output, mean, std)
            #print("Output after moex : ", cls_output)


        #logits = self.classifier(cls_output)
        #print("Logits : ", logits)
        return cls_output
    
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
                input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'], temp_scaling=temp_scaling)
            results = torch.argmax(logits, dim=1)
            prob = F.softmax(logits.to('cpu'), dim=1)
            probabilities += list(prob)

            crt_loss += criterion(logits, x['lbl']
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
                y_pred_unlbl.extend(pred.cpu().numpy())

        y_pred_unlbl = np.array(y_pred_unlbl)
        y_pred_unlbl = np.argmax(y_pred_unlbl, axis=-1).flatten()

    pseudolabeled_data = Dataset_tracked(ds_unlabeled.text_list, y_pred_unlbl, ds_unlabeled.idxes, ds_unlabeled.tokenizer, labeled=True)
    return pseudolabeled_data
    
# Helper function to for AUM calculation during the training dynamics 
def train_ssl_with_aum(pt_teacher_checkpoint, ds_train, ds_pseudolabeled, ulb_epochs, aum_calculator, sup_batch_size):

    model = MoexModel(pt_teacher_checkpoint, num_labels=11)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
    model.train()

    loss_fn_supervised = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_fn_unsupervised = torch.nn.CrossEntropyLoss(reduction='none')

    data_sampler = torch.utils.data.RandomSampler(
        ds_train, num_samples=10**4)
    batch_sampler = torch.utils.data.BatchSampler(
        data_sampler, sup_batch_size, drop_last=False)
    train_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_sampler=batch_sampler)
    
    data_loader_unlabeled = torch.utils.data.DataLoader(ds_pseudolabeled, batch_size=128, shuffle=False) 

    for epoch in range(ulb_epochs):
        for data_supervised, data_unsupervised in tqdm(zip(train_dataloader, data_loader_unlabeled)):
            cuda_tensors_supervised = {key: data_supervised[key].to(
                device) for key in data_supervised if key not in ['idx']}

            cuda_tensors_unsupervised = {key: data_unsupervised[key].to(
                device) for key in data_unsupervised if key not in ['idx']}

            merged_tensors = {}
            for k in cuda_tensors_supervised:
                merged_tensors[k] = torch.cat(
                    (cuda_tensors_supervised[k], cuda_tensors_unsupervised[k]))

            num_lb = cuda_tensors_supervised['input_ids'].shape[0]

            optimizer.zero_grad()
            
            logits_lbls = model(input_ids=cuda_tensors_supervised['input_ids'], token_type_ids=cuda_tensors_supervised[
                'token_type_ids'], attention_mask=cuda_tensors_supervised['attention_mask'])
            logits_ulbl = model(input_ids=cuda_tensors_unsupervised['input_ids'], token_type_ids=cuda_tensors_unsupervised[
                'token_type_ids'], attention_mask=cuda_tensors_unsupervised['attention_mask'])

            aum_calculator.update(logits_ulbl.detach(), cuda_tensors_unsupervised['lbl'], data_unsupervised['idx'])

            loss_sup = loss_fn_supervised(
                logits_lbls, cuda_tensors_supervised['lbl'])
            loss_unsup = loss_fn_unsupervised(
                logits_ulbl, cuda_tensors_unsupervised['lbl'])
            loss = 0.5 * loss_sup + 0.5 * torch.mean(loss_unsup)
            loss.backward()
            optimizer.step()


def train_model_aumst_moex_mixup(hist_train, hist_dev, hist_test, pal_test, pal_unlabeled, pt_teacher_checkpoint, cfg,
                              unsup_epochs=12, lam_moex=0.5,
                    sup_batch_size=32, sup_epochs=10, N_base=10, results_file="results.json", temp_scaling=False, method_type=""):
    
    logger_dict = {}
    logger_dict["Temperature Scaling"] = temp_scaling

    load_best = False

    model = MoexModel(pt_teacher_checkpoint, num_labels=10)
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

            model = MoexModel(pt_teacher_checkpoint)
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
                    loss = criterion_supervised(logits, cuda_tensors['lbl'])
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
    model = MoexModel(pt_teacher_checkpoint)
    model.to(device)
    state_dict = torch.load(main_dir + method_type + "/pytorch_model.bin")
    model.load_state_dict(state_dict)

    for epoch in range(unsup_epochs):

        #-------------AUM calculation------------------
        aum_calculator = AUMCalculator(main_dir + method_type, compressed=False)

        pseudolabled_data = predict_unlabeled(model, pal_unlabeled)

        train_ssl_with_aum(pt_teacher_checkpoint, hist_train, pseudolabled_data, sup_epochs, aum_calculator, sup_batch_size)
        aum_calculator.finalize()
        aum_values_df = pd.read_csv(os.path.join(main_dir + method_type, 'aum_values.csv'))

        aum_values = aum_values_df['aum'].to_list()
        aum_values.sort()
        median_aum_id = int(float(len(aum_values))* 0.5)
        median_aum_value = aum_values[median_aum_id]
        high_aum_ids, low_aum_ids = [], []

        for i, row in aum_values_df.iterrows():
            if row['aum'] > median_aum_value:
                high_aum_ids.append(int(row['sample_id']))
            else:
                low_aum_ids.append(int(row['sample_id']))

        print("Low aum : ", len(low_aum_ids))
        print("High aum : ", len(high_aum_ids))

        low_aum_data = pseudolabled_data.get_subset_dataset(low_aum_ids)
        high_aum_data = pseudolabled_data.get_subset_dataset(high_aum_ids)

        #------------------------ the student model training part with high AUM pseudolabeled samples and moex
        # mixup btw high and low aum samples -----------------------------

        data_sampler = torch.utils.data.RandomSampler(hist_train, num_samples=10**4)
        batch_sampler = torch.utils.data.BatchSampler(data_sampler, sup_batch_size, drop_last=False)
        train_dataloader = torch.utils.data.DataLoader(hist_train, batch_sampler=batch_sampler)
        data_loader_unlabeled_low = torch.utils.data.DataLoader(low_aum_data, batch_size=64, shuffle=False)
        data_loader_unlabeled_high = torch.utils.data.DataLoader(high_aum_data, batch_size=64, shuffle=False) 

        # resetting these values for each student training iteration
        best_f1_overall = 0
        crt_patience = 0

        for st_epoch in range(sup_epochs): #the student model trains for sup_epochs on the pseudo-labeled data + labeled data
            for data_supervised, data_unsup_low, data_unsup_high in tqdm(zip(train_dataloader, data_loader_unlabeled_low, data_loader_unlabeled_high)):
                cuda_tensors_supervised = {key: data_supervised[key].to(device) for key in data_supervised if key not in ['idx']}
                
                cuda_tensors_unsup_low = {key: data_unsup_low[key].to(device) for key in data_unsup_low if key not in ['idx']}

                cuda_tensors_unsup_high = {key: data_unsup_high[key].to(device) for key in data_unsup_high if key not in ['idx']}

                optimizer.zero_grad()

                logits_lbls = model(input_ids=cuda_tensors_supervised['input_ids'], 
                                    token_type_ids=cuda_tensors_supervised['token_type_ids'], 
                                    attention_mask=cuda_tensors_supervised['attention_mask']).logits
                logits_ulbl_high = model(input_ids=cuda_tensors_unsup_high['input_ids'], 
                                            token_type_ids=cuda_tensors_unsup_high['token_type_ids'], 
                                            attention_mask=cuda_tensors_unsup_high['attention_mask']).logits
                
                logits_moex_low_high = model(input_ids=cuda_tensors_unsup_low['input_ids'],
                                            token_type_ids=cuda_tensors_unsup_low['token_type_ids'], 
                                            attention_mask=cuda_tensors_unsup_low['attention_mask'],
                                            input_ids2=cuda_tensors_unsup_high['input_ids'], 
                                            token_type_ids2=cuda_tensors_unsup_high['token_type_ids'], 
                                            attention_mask2=cuda_tensors_unsup_high['attention_mask']).logits
                logits_moex_high_low = model(input_ids=cuda_tensors_unsup_high['input_ids'],
                                            token_type_ids=cuda_tensors_unsup_high['token_type_ids'], 
                                            attention_mask=cuda_tensors_unsup_high['attention_mask'],
                                            input_ids2=cuda_tensors_unsup_low['input_ids'], 
                                            token_type_ids2=cuda_tensors_unsup_low['token_type_ids'], 
                                            attention_mask2=cuda_tensors_unsup_low['attention_mask']).logits
                
                loss_sup = criterion_supervised(logits_lbls, cuda_tensors_supervised['lbl'])
                loss_unsup = criterion_unsupervised(logits_ulbl_high, cuda_tensors_unsup_high['lbl'])

                print("getting moex losses")
                loss_moex1 = criterion_supervised(logits_moex_low_high, cuda_tensors_unsup_low['lbl']) * lam_moex + criterion_supervised(logits_moex_low_high, cuda_tensors_unsup_high['lbl']) * (1. - lam_moex)
                loss_moex2 = criterion_supervised(logits_moex_high_low, cuda_tensors_unsup_high['lbl']) * lam_moex + criterion_supervised(logits_moex_high_low, cuda_tensors_unsup_low['lbl']) * (1. - lam_moex)
                loss = 0.4 * loss_sup + 0.4 * torch.mean(loss_unsup) + 0.1 * loss_moex1 + 0.1 * loss_moex2
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
    best_model = MoexModel(pt_teacher_checkpoint)
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
  
                result = best_model(cuda_tensors['input_ids'], cuda_tensors['token_type_ids'], cuda_tensors['attention_mask'], temp_scaling=True)

                loss = criterion_supervised(result, cuda_tensors['lbl'])
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
