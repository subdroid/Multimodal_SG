from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM, set_seed
import torch
from utils import list_sentences, autoregressive_sentence_formatting, save_json, load_logs_mmsg_conditionwise
from text_generation import generate
import os
import json
import torch
from torch.nn import functional as F
import re
from collections import defaultdict
import numpy as np
import shutil 
import pandas as pd
import numpy as np
 

def aggregate_particpant_stats(condition):
    
    """
    
    Paticipants merely consists of the list of UIDs.

    Sentences consist of a list of sentences.

    Accuracy and Confidence are lists extracted from the collected data participant wise
    corresponding to the specified condition.
    
    """
    
    Participants, sentences, Accuracy, Confidence = load_logs_mmsg_conditionwise(condition)

   
    DataFrames_Acc  = [[] for i in range(len(sentences))] #List of list containing accuracy data for a sentence
    DataFrames_Conf = [[] for i in range(len(sentences))] #List of list containing confidence data for a sentence

    for pid in Participants:
        for item in Accuracy:
            uid      = item['uid']
            sent     = item['sent']
            w_times  = item['word_times']
            accuracy = item['accuracy']
            if uid==pid:
                for s_id in range(len(sentences)):
                    if sent==sentences[s_id]:
                        DataFrames_Acc[s_id].append(accuracy)
        for item in Confidence:
            uid        = item['uid']
            sent       = item['sent']
            w_times    = item['word_times']
            confidence = item['confidence']
            if uid==pid:
                for s_id in range(len(sentences)):
                    if sent==sentences[s_id]:
                        DataFrames_Conf[s_id].append(confidence)
        
    Acc_HA   = [] #Accuracy list for Human Agreement scoring
    Acc      = []
    Conf_HA  = [] #Confidence list for Human Agreement scoring
    Conf     = []
    
    for items in range(len(DataFrames_Acc)):
        data_acc = DataFrames_Acc[items]
        sent     = sentences[items]
        sent     = re.sub(r'[^\w\s]', '', sent)
        l_sent   = len(sent.split())
        n_items  = len(data_acc)
        ct = 0

        tmp = [] #temporary store for data list
        for col in data_acc:
            if len(col)==l_sent:
                tmp.append(col)

        if len(tmp)>=1:
            dt_tmp             = {}
            dt_tmp['Sentence'] = [sent for i in range(l_sent)] #put sentence on each row of the DataFrame
            df                 = pd.DataFrame(dt_tmp)
            for col in tmp:
                ct   += 1
                nm    = "subject"+str(ct)
                df[nm]= col
            Acc.append(df)
            if len(tmp)>=2:
                Acc_HA.append(df) 
    
    for items in range(len(DataFrames_Conf)):
        data_conf = DataFrames_Conf[items]
        sent     = sentences[items]
        sent     = re.sub(r'[^\w\s]', '', sent)
        l_sent   = len(sent.split())
        n_items  = len(data_conf)
        ct = 0

        tmp = [] #temporary store for data list
        for col in data_conf:
            if len(col)==l_sent:
                tmp.append(col)

        if len(tmp)>=1:
            dt_tmp             = {}
            dt_tmp['Sentence'] = [sent for i in range(l_sent)] #put sentence on each row of the DataFrame
            df                 = pd.DataFrame(dt_tmp)
            for col in tmp:
                ct   += 1
                nm    = "subject"+str(ct)
                df[nm]= col
            Conf.append(df)
            if len(tmp)>=2:
                Conf_HA.append(df) 
           
    return Acc_HA, Conf_HA, Acc, Conf