#!/usr/bin/env python3
import os
import json
import re
import shutil 
import pandas as pd
import numpy as np
from utils import read_json

def process_labels(labels):
    label = []
    
    for l in labels:
        txt = l.split()[:-1]
        txt = (' '.join([str(el) for el in txt])).strip()
        if txt not in label:
            label.append(txt)
    
    Labels = ' , '.join([str(el) for el in label])
    
    return Labels


def read_sent_labels(f_name):
    
    data       = read_json(f_name)
    sentence   = data["caption"]
    labels     = data["labels"]
    Labels     = process_labels(labels)
    txt_only   = "<|startoftext|>  "+sentence
    multimodal = Labels + " <|startoftext|>  "+sentence
    
    return txt_only.strip(), multimodal.strip()

def read_data():
    file_doc1 = os.path.join(os.path.dirname(os.getcwd()),os.path.join("data","labels_text_onlytext"))
    file_doc2 = os.path.join(os.path.dirname(os.getcwd()),os.path.join("data","labels_text"))
    
    loc = os.path.join(os.path.dirname(os.getcwd()),os.path.join("data","v1_img"))

    f1_ = open(file_doc1,"w")
    f2_ = open(file_doc2,"w")
    
    for folder in os.listdir(loc):
        for files in os.listdir(os.path.join(loc,folder)):
            f_type = files.split(".")[-1]
            if f_type=="json":
                txt_only, multimodal = read_sent_labels(os.path.join(loc,os.path.join(folder,files)))
                print(txt_only,   file=f1_)
                print(multimodal, file=f2_)
    
    f1_.close()
    f2_.close()

read_data()