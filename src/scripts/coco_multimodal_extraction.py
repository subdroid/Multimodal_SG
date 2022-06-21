#!/bin/python 

#Extract the COCO captions for finetuning PTLMs. 

import os
from utils import read_json
import re
import random
import wget
import subprocess
from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
import requests
import torch


def get_object_labels(image):
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    inputs = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]
    
def multimodal_data_maker(t_data,image_path,coco_master_file,filename1,filename2):
    
    data_loc             = os.path.join(os.path.dirname(os.getcwd()),"data")
    coco_master    = read_json(coco_master_file)
    
    image_dic = {}
    t_images = coco_master["images"]
    t_annotations = coco_master["annotations"]

    img_loc = os.path.join(data_loc,image_path)
    

    for images in t_images:
        image_url = images["coco_url"]
        image_name = image_url.split("/")[-1]
        image_id = images["id"]
        image_dic[image_id] = os.path.join(img_loc,image_name)
    
    
    
    file1_=open(os.path.join(data_loc,filename1),"w")
    file2_=open(os.path.join(data_loc,filename2),"w")

    for instance in t_annotations:
        caption  = instance["caption"]
        id_image = instance["image_id"]
        img_file = image_dic[id_image]
        if caption in t_data:
            if image_dic[id_image]:
                image_  = image_dic[id_image]
                image   = Image.open(image_dic[id_image])
                try:
                    objects = get_object_labels(image)
                    item_whole = objects + " <startoftext> " + caption
                    print(item_whole,file=file1_)
                    cap_part = caption.split()
                    for i in range(len(cap_part)):
                        item_part = objects + " <startoftext> "+ ' '.join(cap_part[:i+1])
                        print(item_part,file=file2_)
                except :
                    continue
                # break
            


def get_images_list():
    data_loc             = os.path.join(os.path.dirname(os.getcwd()),"data")
    
    data_text_train      = os.path.join(data_loc,"coco_caption_20percent")
    data_train           = open(data_text_train,"r").read().split("\n") 
   
    train_data = []
    for sent in data_train:
        if sent:
            train_data.append((sent.replace("<startoftext>","")).strip())
    
    image_path           = "train2014"
    coco_cap_master_file = os.path.join(data_loc,os.path.join("annotations","captions_train2014.json"))
    filename1 = "coco_multimodal_train_20percent"
    filename2 = "coco_multimodal_train_20percent_curr"
    multimodal_data_maker(train_data,image_path,coco_cap_master_file,filename1,filename2)  

    # FULL Training Data
    data_text_train      = os.path.join(data_loc,"coco_caption_full")
    data_train           = open(data_text_train,"r").read().split("\n") 
   
    train_data = []
    for sent in data_train:
        if sent:
            train_data.append((sent.replace("<startoftext>","")).strip())
    
    image_path           = "train2014"
    coco_cap_master_file = os.path.join(data_loc,os.path.join("annotations","captions_train2014.json"))
    filename1 = "coco_multimodal_train_full"
    filename2 = "coco_multimodal_train_full_curr"
    multimodal_data_maker(train_data,image_path,coco_cap_master_file,filename1,filename2)  


    
   
   
    data_text_test      = os.path.join(data_loc,"coco_val_01percent")
    data_test           = open(data_text_train,"r").read().split("\n") 
    test_data  = []
    for sent in data_test:
        if sent:
            test_data.append((sent.replace("<startoftext>","")).strip())
    
    image_path           = "test2014"
    coco_cap_master_file = os.path.join(data_loc,os.path.join("annotations","captions_val2014.json"))
    filename1 = "coco_multimodal_val_01percent"
    filename2 = "coco_multimodal_val_01percent_curr"
    multimodal_data_maker(test_data,image_path,coco_cap_master_file,filename1,filename2)  
    
        

def get_data():
    
    data_loc            = os.path.join(os.path.dirname(os.getcwd()),"data")
    
    coco_cap_train      = os.path.join("annotations","captions_train2014.json")
    COCO_Captions_train = os.path.join(data_loc,coco_cap_train)

    coco_cap_val        = os.path.join("annotations","captions_val2014.json")
    COCO_Captions_val   = os.path.join(data_loc,coco_cap_val)


    COCO = []

    coco_train       = read_json(COCO_Captions_train)
    annotations_coco = coco_train["annotations"]
    annot_loc        = os.path.join(data_loc,"coco_train_annotations")
    annot_coco       = open(annot_loc,"w")

    for instances in annotations_coco:
        
        caption = instances["caption"]
        
        if caption not in COCO:
            caption = (re.sub(r'[^\w\s]', '', caption)).strip()+"."
            COCO.append(caption)
            print(caption,file=annot_coco)
    
    COCO = []
    
    coco_val         = read_json(COCO_Captions_val)
    annotations_coco = coco_val["annotations"]
    annot_loc        = os.path.join(data_loc,"coco_val_annotations")
    annot_coco       = open(annot_loc,"w")

    for instances in annotations_coco:
        
        caption = instances["caption"]
        
        if caption not in COCO:
            caption = (re.sub(r'[^\w\s]', '', caption)).strip()+"."
            COCO.append(caption)
            print(caption,file=annot_coco)


def sample_data(dataset,n_perc):
    data    = open(dataset,"r").read().split("\n")
    n_lines = int(len(data)*n_perc)
    random.shuffle(data)
    return random.sample(data, n_lines)


def save_sample_file(data,f_name):
    data_loc       = os.path.join(os.path.dirname(os.getcwd()),"data")
    file_location  = os.path.join(data_loc,f_name)
    file2_location = os.path.join(data_loc,f_name+"_curr")
     
    file1_ = open(file_location,"w")
    file2_ = open(file2_location,"w")
    for line in data:
        txt = "<startoftext> "+line
        print(txt,file=file1_)
        l_split = line.split()
        for i in range(len(l_split)):
            print("<startoftext> "+' '.join(l_split[:i+1]),file=file2_)
    


# get_data()

data_loc            = os.path.join(os.path.dirname(os.getcwd()),"data")
data_location       = os.path.join(data_loc,"coco_train_annotations")

## USING THESE
"""
sampled_data = sample_data(data_location,0.2)
save_sample_file(sampled_data,"coco_caption_20percent")

sampled_data = sample_data(data_location,1.0)
save_sample_file(sampled_data,"coco_caption_full")

data_location      = os.path.join(data_loc,"coco_val_annotations")
sampled_data       = sample_data(data_location,0.01)
save_sample_file(sampled_data,"coco_val_01percent")
"""
get_images_list()