#!/bin/python3

import os
import math
import matplotlib.pyplot as plt 

def load_data(model,dataset,config):
    f_name = model+"_"+dataset+"_"+config
    data_loc = os.path.join(os.getcwd(),"training_logs")
    #gpt_base_coco20_full_finetune
    data_file = open(os.path.join(data_loc,f_name),"r").read().split("\n")
    print(data_file)


load_data("gpt_base","coco20","full_finetune")