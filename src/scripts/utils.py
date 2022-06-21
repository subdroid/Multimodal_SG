import re
from collections import defaultdict
import pickle
import json
import os
from pathlib import Path
from glob import glob
import numpy as np

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def read_json(path):
    with open(path, "r") as fread:
        return json.load(fread)

def load_logs_mmsg_conditionwise(condition):
    root_loc = Path(os.getcwd()).parent
    data_loc = os.path.join(root_loc,os.path.join("data","mmsg"))
    uids     = glob(os.path.join(data_loc,"*.jsonl"))
    file_loc = os.path.join(root_loc,os.path.join("computed","human_"+condition))
    file_    = open(file_loc,"w")
    
    Sentences    = []
    sentences    = [] #list of sentences appearing for the condition
    ESentences   = []
    Participants = []
    
    #loop through all the files (user ids)
    for uid in uids:
         with open(f"{uid}", "r") as f:
            
            name  = uid.split("/")[-1].replace(".jsonl","")
            Participants.append(name)
            
            data_ = [{**json.loads(line)}for line in f.readlines()]
            for item in range(len(data_)):

                config = data_[item]['config']
                id_    = data_[item]['id']

                if config==condition:

                    Word_times_accuracy   = []
                    Accuracy              = []
                    Word_times_confidence = []
                    Confidence            = []
             
                    sent = data_[item]['sent']

                    if sent not in sentences:
                        sentences.append(sent)
                    
                    ratings = data_[item]['ratings'] 
             
                    for word in ratings:
                        # the first tuple represents particpants's confidence about predicting the next word
                        interest_tuple = word[0]
                        Word_times_confidence.append(interest_tuple[0])
                        Confidence.append(interest_tuple[1])
                        # the first tuple represents particpants's accuracy of predicting the word
                        interest_tuple = word[1]
                        Word_times_accuracy.append(interest_tuple[0])
                        Accuracy.append(interest_tuple[1])
                        

                    sentence  = {"uid":name,"sent":sent,"word_times":Word_times_accuracy,"accuracy":Accuracy,"id":id_}
                    Esentence = {"uid":name,"sent":sent,"word_times":Word_times_confidence,"confidence":Confidence,"id":id_}
                    
                    Sentences.append(sentence)
                    ESentences.append(Esentence)
    
    for instance1,instance2 in zip(Sentences,ESentences):
        print(instance1,file=file_)
        print(instance2,file=file_)
        print("\n",file=file_)

    return Participants, sentences, Sentences, ESentences
                
def autoregressive_sentence_formatting(sentence):
    sent = sentence.split()
    sent_temp = ''
    Part_sent = []
    for index in range(1,len(sent)):
        sent_temp = sent[:index+1]
        temp = sent_temp[-1]
        
        sent_temp[-1] = ''

        part_sent = (' '.join(word for word in sent_temp)).strip()
        Part_sent.append(part_sent)
        sent_temp[-1] = temp
    return Part_sent

def list_sentences(config):
    computed_path = os.path.join(os.path.dirname(os.getcwd()),"computed")
    f_name = "human_data_accuracy_"+config
    data_file = os.path.join(computed_path,f_name)
    content = read_json(data_file)
    
    sentences = []
    
    for sentence in content:
        s = sentence["sent"]
        if s not in sentences:
            sentences.append(s)
    
    return sentences

