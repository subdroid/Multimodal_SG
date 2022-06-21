#!/usr/bin/env python3
import re
import glob
from utils import read_json, load_logs_mmsg_conditionwise
import os 
import json
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

def aggregate_particpant_stats(condition):
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
 
def print_HUMAN_agreement_stats(computed_path,Accuracy,Conf):
    
    """
    Accuracy and Conf are list of Pandas Dataframes. 
    Every element of the list(s) corresponds to a particular sentence.
    The columns of the DataFrame represents the different participant performances.  
    """

    file_   = open(os.path.join(computed_path,"HUMAN_AGREEMENT"),"w")
    
    Avg_Acc = []
    
    print("Accuracy\n\n",file=file_)
    
    for sent in Accuracy:
        data             = sent
        cols             = list(sent.columns)
        sentence         = sent["Sentence"][0]
        print(sentence,file=file_)
        
        cols.remove("Sentence")
        results,mean,std = compare_human_stats(data,cols)

        
        print(results,file=file_)
        if len(cols)>2:
            print("Mean\t%f"%mean,file=file_)
            print("Std\t%f"%std,file=file_)
        print("\n\n",file=file_)
        
        Avg_Acc.append(mean) 
    
    print("Overall mean:%f\tstd:%f"%(np.mean(Avg_Acc),np.std(Avg_Acc)),file=file_)

    Avg_Conf = []

    print("\n\nConfidence\n\n",file=file_)

    for sent in Conf:
        data             = sent
        cols             = list(sent.columns)
        sentence         = sent["Sentence"][0]
        print(sentence,file=file_)
        
        cols.remove("Sentence")
        results,mean,std = compare_human_stats(data,cols)
        
        print(results,file=file_)
        if len(cols)>2:
            print("Mean\t%f"%mean,file=file_)
            print("Std\t%f"%std,file=file_)
        print("\n\n",file=file_)   
        
        Avg_Conf.append(mean) 
     
    print("Overall mean:%f\tstd:%f"%(np.mean(Avg_Conf),np.std(Avg_Conf)),file=file_)


def compare_human_stats(Data_Matrix,column_names):
    
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


def read_machine_data(path,file_name,condition,sentences):
    
    machine_data    = os.path.join(path,file_name)
    content         = read_json(machine_data)
    
    Accuracy        = []
    Acc_confidence  = []
    Pred_confidence = []
    Acc_score       = []
    Pred_score      = []

    for sentence in sentences:
        sent                  = re.sub(r'[^\w\s]', '', sentence)
        
        accuracy_confidence   = np.array(content[sent]["accuracy_confidence"])[:,1]
        prediction_confidence = np.array(content[sent]["prediction_confidence"])[:,1]
        accuracy_score        = np.array(content[sent]["accuracy_score"])[:,1]
        prediction_score      = np.array(content[sent]["prediction_score"])[:,1]
        score                 = content[sent]["score"]
        
        AC = []
        PC = []
        AS = []
        PS = []
                
        for item in accuracy_confidence:
            AC.append(float(item))
        
        for item in prediction_confidence:
            PC.append(float(item))
        
        for item in accuracy_score:
            AS.append(float(item))
        
        for item in prediction_score:
            PS.append(float(item))

        Accuracy.append(score[0])
        Acc_confidence.append(AC) 
        Pred_confidence.append(PC)
        Acc_score.append(AS) 
        Pred_score.append(PS)
        
    return Accuracy, Acc_confidence, Pred_confidence, Acc_score, Pred_score

def compare_human_machine_stats(Data_Matrix,column_names,model_name):

    correlations = {}
    
    for col_a, col_b in itertools.combinations(column_names, 2):
        if col_a==model_name or col_b==model_name:
            el1 = Data_Matrix.loc[:, col_a]
            el2 = Data_Matrix.loc[:, col_b]
            els1 = np.unique(el1)
            els2 = np.unique(el2)
            # This is to fix a bug that emerged with certain observations
            if len(np.unique(el1))>=2 and len(np.unique(el2))>=2:
                correlations[col_a + '_' + col_b] = pearsonr(el1, el2)           
            
    result         = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']

    sig_results = result.loc[result['p-value'] <= 0.05]

    pcc     = (result['PCC']).tolist()
    pcc_sig = (sig_results['PCC']).tolist()

    if len(pcc)>1:
        mean = np.mean(pcc)
        std = np.std(pcc)
    else:
        mean = pcc[0]
        std = 0.0
    
    mean_sig = 0.0
    std_sig = 0.0
    if len(pcc_sig)>0:
        if len(pcc_sig)>1:
            mean_sig = np.mean(pcc_sig)
            std_sig = np.std(pcc_sig)
        else:
            mean_sig = pcc_sig[0]
            std_sig = 0.0

    return result.sort_index(),mean,std, sig_results.sort_index(),mean_sig,std_sig

def plot_PCC(data=None,names=None,X_axis=None,condition=None,f_type=None,f_name=None):
    c = 0
    for models in data:
        name = names[c]
        plt.plot(X_axis,models,label=name)
        c+=1
    plt.legend()
    plt.xticks(np.arange(0,len(X_axis), 1.0))
    plt.xlabel("Sentences")
    plt.ylabel("Pearson Correlation Coefficient")
    ttl = "Correlation between humans and GPT2: "+condition+" ("+f_type+")"
    plt.title(ttl)
    plt.savefig(f_name)
    plt.clf()
    plt.close()
    

def plot_PCC_hmap(data=None,names=None,X_axis=None,condition=None,f_type=None,f_name=None):
    map_data = np.array(data)
    ax = sns.heatmap(map_data,yticklabels=names,vmin=-1, vmax=1)
    plt.xlabel("Sentences")
    # ttl = "Pearson Correlation between humans and GPT2: "+condition+" ("+f_type+")"
    ttl = "Pearson Correlation between humans and GPT2: "+condition
    plt.title(ttl)
    plt.savefig(f_name)
    plt.clf()
    plt.close()

def equalize_ar(ar):
    n_ar = []

    for model_stats in ar:
        tmp   = []
        nz_id = ((np.nonzero(model_stats))[0])[0]
        
        for e_id in range(nz_id):
            tmp.append(model_stats[nz_id])
        
        tmp = tmp + model_stats[nz_id:]
        n_ar.append(tmp)
   
    return n_ar


computed_path    = os.path.join(os.path.dirname(os.getcwd()),"computed")
conditions       = ["no_image"]
machine_models = ["gpt2_base","gpt2_medium","gpt2_large"]


for condition in conditions:
    print("current condition: %s"%condition)
    Model_Accuracy             = []
    Model_Acc_Confidence       = []
    Model_Pred_Confidence      = []
    """
    Accuracy_Human corresponds to the 
    """
    Accuracy_Human, Conf_Human, Accuracy, Conf = aggregate_particpant_stats(condition)

    # print_HUMAN_agreement_stats(computed_path,Accuracy_Human,Conf_Human)    
    
    Acc_Prob    = []
    Acc_Score   = []
    Conf_Prob   = []
    Conf_Score  = []

    s_Acc_Prob    = []
    s_Acc_Score   = []
    s_Conf_Prob   = []
    s_Conf_Score  = []


    for model in machine_models:
        
        print(model)

        file1_       = open(os.path.join(computed_path,"Human_"+model+"_Prob"),"w")
        file2_       = open(os.path.join(computed_path,"Human_"+model+"_Score"),"w")
        cond        = model+"_"+condition
        sentences   = list_sentences(Accuracy)
        X_axis = np.arange(len(sentences))

        Accuracy_machine, Acc_Conf_machine, Pred_Conf_machine, Acc_score_machine, Pred_score_machine  = read_machine_data(computed_path,cond,condition,sentences)
        
        Avg_Acc_prob  = []
        Avg_Acc_score = []
        s_Avg_Acc_prob  = []
        s_Avg_Acc_score = []
        
        
        print("Accuracy\n\n",file=file1_)
        print("Accuracy\n\n",file=file2_)

        s_mean_hist = 0
        
        for df_id in range(len(Accuracy)):
            sentence = sentences[df_id]

            print(sentence,file=file1_)
            print(sentence,file=file2_)            
            
            comp_matrix       = Accuracy[df_id]
            machine_data_prob = pd.DataFrame(Acc_Conf_machine[df_id],columns=[model])
            comp_matrix       = pd.concat([comp_matrix, machine_data_prob], axis=1)
            
            cols = list(comp_matrix.columns)
            cols.remove("Sentence")

            print(comp_matrix,file=file1_)
            print("\n",file=file1_) 
            
            results, mean, std, s_results, s_mean, s_std  = compare_human_machine_stats(comp_matrix,cols,model)
            
            print(results,file=file1_)
            print("Mean\t%f"%mean,file=file1_)
            print("Std\t%f"%std,file=file1_)
            Avg_Acc_prob.append(mean) 

            if not s_results.empty:
                print("\n\n",file=file1_)
                print(s_results,file=file1_)
                print("Mean\t%f"%s_mean,file=file1_)
                print("Std\t%f"%s_std,file=file1_)
                s_mean_hist = s_mean
            
            s_Avg_Acc_prob.append(s_mean_hist) 
            
            print("\n\n",file=file1_)  

            comp_matrix        = Accuracy[df_id]
            machine_data_score = pd.DataFrame(Acc_score_machine[df_id],columns=[model])
            comp_matrix        = pd.concat([comp_matrix, machine_data_score], axis=1)
            
            cols = list(comp_matrix.columns)
            cols.remove("Sentence")

            print(comp_matrix,file=file2_)
            print("\n",file=file2_) 
            
            results,mean,std, s_results, s_mean, s_std = compare_human_machine_stats(comp_matrix,cols,model)
            
            print(results,file=file2_)
            print("Mean\t%f"%mean,file=file2_)
            print("Std\t%f"%std,file=file2_)
            Avg_Acc_score.append(mean)

            if not s_results.empty:
                print("\n\n",file=file2_)
                print(s_results,file=file2_)
                print("Mean\t%f"%s_mean,file=file2_)
                print("Std\t%f"%s_std,file=file2_)
                s_mean_hist = s_mean
            
            s_Avg_Acc_score.append(s_mean_hist)


            print("\n\n",file=file2_)  
        
        print("Overall mean:%f\tstd:%f"%(np.mean(Avg_Acc_prob),np.std(Avg_Acc_prob)),file=file1_)
        print("Overall signigficant mean:%f\tstd:%f"%(np.mean(s_Avg_Acc_prob),np.std(s_Avg_Acc_prob)),file=file1_)
                 
        print("Overall mean:%f\tstd:%f"%(np.mean(Avg_Acc_score),np.std(Avg_Acc_score)),file=file2_)
        print("Overall significant mean:%f\tstd:%f"%(np.mean(s_Avg_Acc_score),np.std(s_Avg_Acc_score)),file=file2_)

        Acc_Prob.append(Avg_Acc_prob)
        Acc_Score.append(Avg_Acc_score)
        s_Acc_Prob.append(s_Avg_Acc_prob)
        s_Acc_Score.append(s_Avg_Acc_score)
        
        
        
        Avg_Conf_prob  = []
        Avg_Conf_score = []
        s_Avg_Conf_prob  = []
        s_Avg_Conf_score = []
        
        print("\n\nConfidence\n\n",file=file1_)
        print("\n\nConfidence\n\n",file=file2_)
        
        s_mean_hist = 0

        for df_id in range(len(Conf)):
            sentence = sentences[df_id]

            print(sentence,file=file1_)
            print(sentence,file=file2_)            
            
            comp_matrix       = Conf[df_id]
            machine_data_prob = pd.DataFrame(Pred_Conf_machine[df_id],columns=[model])
            comp_matrix       = pd.concat([comp_matrix, machine_data_prob], axis=1)
            
            cols = list(comp_matrix.columns)
            cols.remove("Sentence")

            print(comp_matrix,file=file1_)
            print("\n",file=file1_) 
            
            results, mean, std, s_results, s_mean, s_std  = compare_human_machine_stats(comp_matrix,cols,model)
            
            print(results,file=file1_)
            print("Mean\t%f"%mean,file=file1_)
            print("Std\t%f"%std,file=file1_)
            Avg_Conf_prob.append(mean) 
            
            if not s_results.empty:
                print("\n\n",file=file1_)
                print(s_results,file=file1_)
                print("Mean\t%f"%s_mean,file=file1_)
                print("Std\t%f"%s_std,file=file1_)
                s_mean_hist = s_mean
            
            s_Avg_Conf_prob.append(s_mean_hist) 

            print("\n\n",file=file1_)  

            comp_matrix        = Conf[df_id]
            machine_data_score = pd.DataFrame(Pred_score_machine[df_id],columns=[model])
            comp_matrix        = pd.concat([comp_matrix, machine_data_score], axis=1)
            
            cols = list(comp_matrix.columns)
            cols.remove("Sentence")

            print(comp_matrix,file=file2_)
            print("\n",file=file2_) 
            
            results, mean, std, s_results, s_mean, s_std = compare_human_machine_stats(comp_matrix,cols,model)
            
            print(results,file=file2_)            
            print("Mean\t%f"%mean,file=file2_)
            print("Std\t%f"%std,file=file2_)
            Avg_Conf_score.append(mean)

            if not s_results.empty:
                print("\n\n",file=file2_)
                print(s_results,file=file2_)            
                print("Mean\t%f"%s_mean,file=file2_)
                print("Std\t%f"%s_std,file=file2_)
                s_mean_hist = s_mean
            
            s_Avg_Conf_score.append(s_mean_hist) 
            
            print("\n\n",file=file2_)  

        print("Overall mean:%f\tstd:%f"%(np.mean(Avg_Conf_prob),np.std(Avg_Conf_prob)),file=file1_)
        print("Overall significant mean:%f\tstd:%f"%(np.mean(s_Avg_Conf_prob),np.std(s_Avg_Conf_prob)),file=file1_)
        
         
        print("Overall mean:%f\tstd:%f"%(np.mean(Avg_Conf_score),np.std(Avg_Conf_score)),file=file2_)
        print("Overall significant mean:%f\tstd:%f"%(np.mean(s_Avg_Conf_score),np.std(s_Avg_Conf_score)),file=file2_)
        
       
        Conf_Prob.append(Avg_Conf_prob)
        Conf_Score.append(Avg_Conf_score)        
        s_Conf_Prob.append(s_Avg_Conf_prob)
        s_Conf_Score.append(s_Avg_Conf_score)

    # plot_PCC(data=Acc_Prob,   names=machine_models, X_axis=X_axis, condition="accuracy",   f_type="prob",  f_name="h_gpt_acc_prob")
    # plot_PCC(data=Acc_Score,  names=machine_models, X_axis=X_axis, condition="accuracy",   f_type="score", f_name="h_gpt_acc_score")
    # plot_PCC(data=Conf_Prob,  names=machine_models, X_axis=X_axis, condition="confidence", f_type="prob",  f_name="h_gpt_conf_prob")
    # plot_PCC(data=Conf_Score, names=machine_models, X_axis=X_axis, condition="confidence", f_type="score", f_name="h_gpt_conf_score")

    # plot_PCC_hmap(data=Acc_Prob,     names=machine_models, X_axis=X_axis, condition="accuracy",     f_type="prob",    f_name="h_gpt_acc_prob_hmap")
    # plot_PCC_hmap(data=Acc_Score,    names=machine_models, X_axis=X_axis, condition="accuracy",     f_type="score",   f_name="h_gpt_acc_score_hmap")
    # plot_PCC_hmap(data=Conf_Prob,    names=machine_models, X_axis=X_axis, condition="confidence",   f_type="prob",    f_name="h_gpt_conf_prob_hmap")
    # plot_PCC_hmap(data=Conf_Score,   names=machine_models, X_axis=X_axis, condition="confidence",   f_type="score",   f_name="h_gpt_conf_score_hmap")

    
    
    eq_Acc = equalize_ar(s_Acc_Score)
    eq_Conf = equalize_ar(s_Conf_Score)

    
    plot_PCC_hmap(data=Acc_Score,    names=machine_models, X_axis=X_axis, condition="accuracy",     f_type="score",   f_name="h_gpt_acc_hmap")
    plot_PCC_hmap(data=Conf_Score,   names=machine_models, X_axis=X_axis, condition="confidence",   f_type="score",   f_name="h_gpt_conf_hmap")
    plot_PCC_hmap(data=eq_Acc,    names=machine_models, X_axis=X_axis, condition="s_accuracy",     f_type="score",   f_name="S_h_gpt_acc_hmap")
    plot_PCC_hmap(data=eq_Conf,   names=machine_models, X_axis=X_axis, condition="s_confidence",   f_type="score",   f_name="S_h_gpt_conf_hmap")
    