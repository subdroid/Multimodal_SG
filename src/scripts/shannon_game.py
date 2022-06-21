#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, set_seed
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
from msg_utils import aggregate_particpant_stats 

def model_def(model_name): 
    if model_name=="gpt_base":
        tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir="huggingface")
        model     = AutoModelForCausalLM.from_pretrained('gpt2', return_dict_in_generate=True, cache_dir="huggingface")
    
    if model_name=="gpt_medium":
        tokenizer = AutoTokenizer.from_pretrained('gpt2-medium',cache_dir="huggingface")
        model     = AutoModelForCausalLM.from_pretrained('gpt2-medium', return_dict_in_generate=True, cache_dir="huggingface")
    
    if model_name=="gpt_large":
        tokenizer = AutoTokenizer.from_pretrained('gpt2-large',cache_dir="huggingface")
        model     = AutoModelForCausalLM.from_pretrained('gpt2-large', return_dict_in_generate=True, cache_dir="huggingface")
    
    return tokenizer,model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def huggingface_folder_op():
    hug_loc = os.path.join(os.getcwd(),"huggingface")
    shutil.rmtree(hug_loc)
    os.mkdir(hug_loc)
    
    
def list_sentences(Acc):
    sentences = []
    for item in Acc:
        sent = list(item["Sentence"])[0]
        if sent not in sentences:
            sentences.append(sent)
    return sentences

def run_autoregressive_models(model_name,file_name,temperature,sentences,formatted=False):
    
    logdata          = defaultdict(lambda: defaultdict(list))
    tokenizer, model = model_def(model_name)
    stop_flag = False
    
    for sentence in sentences:
        
        if formatted:
            try:
                img_part = sentence.split("<|startoftext|>")[0]
                sentence = sentence.split("<|startoftext|>")[1]
            except:
                stop_flag=True

        if not stop_flag:  
            sent       = re.sub(r'[^\w\s]', '', sentence)
            sent = "<startoftext> "+sent.strip()
            Sent_parts = autoregressive_sentence_formatting(sent)
            
            correct = 0

            for w_id,part_sent in enumerate(Sent_parts):
                if formatted:
                    part_sent = img_part + part_sent

                correct_tok      = " "+(sent.split())[w_id+1]
                correct_token_id = (tokenizer.encode(correct_tok))[0]

                """When for some words where the probability comes as zero, 
                the encoded token (after encoding) is not the same as the original token.
                """
        
                predicted_token,correct_token,confidence,accuracy = generate(model,tokenizer,part_sent,correct_token_id,temperature)
                
               
                if (correct_tok.strip())==(predicted_token.strip()):
                    correct+=1
            
                logdata[sent]["accuracy"].append([correct_tok,accuracy])
                logdata[sent]["confidence"].append([predicted_token,confidence])
                
        logdata[sent]["score"].append(correct/(w_id))

    
    logfile_name=file_name
    print("Saving to", logfile_name)
    save_json(logfile_name, logdata)

def run_machine_experiment(experiment_case,sentences_org=None,sentences_given=False):
    
    print(experiment_case)

    formatted = True

    if experiment_case=="no_image":
        formatted=False

        
    _, _, Accuracy_Human, Conf_Human = aggregate_particpant_stats(experiment_case)
        
    if not sentences_given:
        sentences = list_sentences(Accuracy_Human)

    
    if sentences_given:
        sentences = []
        h_sentences = list_sentences(Accuracy_Human)
        
        for s_el in sentences_org:
            sent = (s_el.split("<|startoftext|>")[-1]).strip()
            sent = re.sub(r'[^\w\s]', '', sent)
            if sent in h_sentences:
                sentences.append(s_el)       
        
    temperature = np.arange(0.1,1.1,0.1)
    f_name      = ["gpt2_base_"+experiment_case,"gpt_medium_"+experiment_case,"gpt_large_"+experiment_case]
    model_names = ["gpt_base","gpt_medium","gpt_large"]
 
    for m_id in range(len(f_name)):
        huggingface_folder_op() #To avoid memory issues

        model_ = model_names[m_id]
        print(model_)
      
        file_name = f_name[m_id] 
        prefix    = "frozen_model"
        fold_loc  = os.path.join(os.path.dirname(os.getcwd()),os.path.join("computed",prefix))

        if not os.path.exists(fold_loc):
            os.mkdir(fold_loc)
    
        for temp in temperature:
            temp = np.around(temp,1)
            t    = "temp_"+str(temp)
            
            fold_loc = os.path.join(os.path.dirname(os.getcwd()),os.path.join("computed",os.path.join(prefix,t)))
            if not os.path.exists(fold_loc):
                os.mkdir(fold_loc)

            f_ = os.path.join(fold_loc,file_name)
            
            """
            Input field details of the arguments for run_autoregressive_models:
            model_ = gpt_<ver> 
            f_ = computed/frozen_model/temp/model_detail_experiment_condition 
            temp = temperature
            """
            
            run_autoregressive_models(model_,f_,temp,sentences,formatted)
    


set_seed(13)

run_machine_experiment("no_image")

labels_only_loc = os.path.join(os.path.join(os.path.dirname(os.getcwd()),"data"),"labels_text")
label_sents = open(labels_only_loc,"r").read().split("\n")
run_machine_experiment("labels_text",label_sents,sentences_given=True)






