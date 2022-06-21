#!/usr/bin/env python3
import re
import glob
from utils import read_json, load_logs_mmsg_conditionwise,save_json
import os 
import json
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from collections import defaultdict
import scipy.stats as st


def aggregate_particpant_stats(condition):
    Participants, sentences, Accuracy, Confidence = load_logs_mmsg_conditionwise(condition)
   
    DataFrames_Acc  = [[] for i in range(len(sentences))] #List of list containing accuracy data for a sentence
    DataFrames_Conf = [[] for i in range(len(sentences))] #List of list containing confidence data for a sentence

    ID = {}

    for pid in Participants:
        for item in Accuracy:
            uid      = item['uid']
            sent     = item['sent']
            w_times  = item['word_times']
            accuracy = item['accuracy']
            id_      = item['id']
            if uid==pid:
                for s_id in range(len(sentences)):
                    if sent==sentences[s_id]:
                        sent = re.sub(r'[^\w\s]', '', sent)
                        DataFrames_Acc[s_id].append(accuracy)
                        if sent not in ID.keys():
                            ID[sent] = id_
        for item in Confidence:
            uid        = item['uid']
            sent       = item['sent']
            w_times    = item['word_times']
            confidence = item['confidence']
            id_      = item['id']
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
        # n_items  = len(data_acc)
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
        # n_items  = len(data_conf)
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
    
    print(ID)
           
    return Acc_HA, Conf_HA, Acc, Conf, ID
 

    
    correlations = {}
    
    for col_a, col_b in itertools.combinations(column_names, 2):
        correlations[col_a + '_' + col_b] = pearsonr(Data_Matrix.loc[:, col_a], Data_Matrix.loc[:, col_b])

    result         = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']

    mean           = np.mean(result['PCC'])
    std            = np.std(result['PCC'])
    
    return result.sort_index(),mean,std

def list_sentences(Acc):
    
    sentences = []
    
    for item in Acc:
        sent  = list(item["Sentence"])[0]
        sentences.append(sent)
    
    return sentences


def read_machine_data(file_name):
    
    machine_data    = os.path.join(file_name)
    content         = read_json(machine_data)

    sentences = content.keys()
    
    Confidence = []
    Accuracy = []
    Correct = []

    Sentences = []
    
    for sentence in sentences:
        Sentence = (sentence.split("<startoftext>")[1] ).strip()
        Sentences.append(Sentence)

        accuracy = np.array(content[sentence]["accuracy"])[:,1]
        confidence = np.array(content[sentence]["confidence"])[:,1]
        score = np.array(content[sentence]["score"])
        Acc = []
        Conf = []
        for item in accuracy:
            Acc.append(float(item))
        for item in confidence:
            Conf.append(float(item))
        Accuracy.append(Acc)
        Confidence.append(Conf)
        Correct.append(score[0])

    return Accuracy, Confidence, Correct, Sentences

def compare_human_machine_stats(Data_Matrix,column_names,model_name):

    correlations = {}
    
    for col_a, col_b in itertools.combinations(column_names, 2):
        if col_a==model_name or col_b==model_name:
            el1 = Data_Matrix.loc[:, col_a]
            el2 = Data_Matrix.loc[:, col_b]
            # This is to fix a bug that emerged with certain observations
            if len(np.unique(el1))>=2 and len(np.unique(el2))>=2:
                correlations[col_a + '_' + col_b] = pearsonr(el1, el2)    

            
    result         = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']
    mean_PCC = result["PCC"].mean()


    return result.sort_index(),mean_PCC



def calc_means(data):
    Avg = []
    for row in data:
        mean_row = np.mean(row)
        Avg.append(mean_row)
    return Avg


def confidence(vals):
    return st.t.interval(
        alpha=0.95,
        df=len(vals) - 1,
        loc=np.mean(vals),
        scale=st.sem(vals)
    )


computed_path    = os.path.join(os.path.dirname(os.getcwd()),"computed")
computed_log     = open(os.path.join(computed_path,"human_machine_log"),"w")
frozen_models    = os.path.join(computed_path,"frozen_model")
conditions       = ["no_image","labels_text"]
machine_models = ["gpt2_base","gpt2_medium","gpt2_large"]

Correctness = defaultdict(dict)



for folders in os.listdir(frozen_models):
    # print("\n\n\ntemperature:\t%s"%folders)
    fold_loc = os.path.join(frozen_models,folders)

    data_def  = defaultdict(lambda: defaultdict(list))
   

    for models in os.listdir(fold_loc):
        # data_def  = defaultdict(lambda: defaultdict(list))
        condition = models.split("_")[-2:]
        model_name = models.split("_")[:-2]

        model_name = '_'.join(model_name)
        condition = '_'.join(condition)      

        Accuracy_Human, Conf_Human, Accuracy, Conf, ID = aggregate_particpant_stats(condition)

        Machine_Accuracy, Machine_Confidence, Machine_Correct, Machine_Sentences = read_machine_data(os.path.join(fold_loc,models))

        for s_id in range(len(Machine_Accuracy)):
            sent     = re.sub(r'[^\w\s]', '', Machine_Sentences[s_id])
            if ID[sent]:
                id_ = ID[sent]
                Ratings = []
                for acc,conf in zip(Machine_Accuracy[s_id],Machine_Confidence[s_id]):
                    Ratings.append([["null",conf],["null",acc]])
                config = condition
                data_def[folders][model_name].append({"sent":sent,"time":'null',"config":config,"id":id_,"ratings":Ratings})
    print(data_def)
    logfile_name = os.path.join(computed_path,"machine_stats.json")
    print("Saving to", logfile_name)
    save_json(logfile_name, data_def)

        

