#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, set_seed, Trainer
from transformers import TextDataset,DataCollatorForLanguageModeling, HfArgumentParser, TrainingArguments
from transformers import AutoTokenizer
import os
from datasets import load_metric
from lm_utils import load_data
import torch
import sys
import logging
import math
import shutil
from utils import save_json

def model_def(model_name): 
    if model_name=="gpt_base":
        tokenizer = AutoTokenizer.from_pretrained('gpt2',cache_dir="huggingface")
        model     = AutoModelForCausalLM.from_pretrained('gpt2',cache_dir="huggingface")
    
    if model_name=="gpt_medium":
        tokenizer = AutoTokenizer.from_pretrained('gpt2-medium',cache_dir="huggingface")
        model     = AutoModelForCausalLM.from_pretrained('gpt2-medium',cache_dir="huggingface")
    
    if model_name=="gpt_large":
        tokenizer = AutoTokenizer.from_pretrained('gpt2-large',cache_dir="huggingface")
        model     = AutoModelForCausalLM.from_pretrained('gpt2-large',cache_dir="huggingface")
    
    return tokenizer,model

def huggingface_folder_op():
    hug_loc = os.path.join(os.getcwd(),"huggingface")
    shutil.rmtree(hug_loc)
    os.mkdir(hug_loc)


def main():

    data_loc              = os.path.join(os.path.dirname(os.getcwd()),"data")

    data_location_01      = os.path.join(data_loc,"coco_caption_01percent")
    data_location_05      = os.path.join(data_loc,"coco_caption_05percent")
    data_location_10      = os.path.join(data_loc,"coco_caption_10percent")
    data_location_100     = os.path.join(data_loc,"coco_val_annotations")

    test_data             = os.path.join(data_loc,"coco_val_01percent")

    model_names = ["gpt_base","gpt_medium","gpt_large"]
    datasets    = [data_location_01,data_location_05,data_location_10,data_location_100]
    log_names   = ["coco01","coco05","coco10","coco100"]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    
    for model_n in model_names:
        
        c_d = 0
        
        for dataset in datasets:
        
            # huggingface_folder_op() #To avoid memory issues
            fold_name = model_n+"_"+log_names[c_d]+"_lnorm_finetune"
            
            log_loc = os.path.join(os.getcwd(),"training_logs")
            logfile_name = os.path.join(log_loc,fold_name+".json")
            log_cont = open(logfile_name,"w")
            log_cont.close()

            logger = logging.getLogger(fold_name)
            logger.setLevel(logging.INFO)

            fold_name = "model_"+fold_name
            
            c_d+=1
            
            training_args = TrainingArguments(fold_name,
            evaluation_strategy = "steps",
            load_best_model_at_end=True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=10,
            save_strategy="steps",
            save_steps=10,
            logging_steps=10,
            logging_first_step=True,
            eval_steps=10,
            save_total_limit=1,
            data_seed=69,
            run_name=fold_name,
            warmup_steps=10,
            prediction_loss_only=True, #for perplexity calculation
            learning_rate = 5e-5,
            weight_decay=0,
            adam_epsilon = 1e-8,
            max_grad_norm = 1.0,
            seed=123)

            tokenizer, model = model_def(model_n)
            tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
            model.resize_token_embeddings(len(tokenizer))
            model.to(device) 
            
            freeze_flag = True
            for name, param in model.named_parameters(): 
                if name!="transformer.wte.weight" and name!="transformer.wpe.weight":
                    if "ln" in name or "mlp" in name:
                        freeze_flag = False
                    if "ln" not in name and "mlp" not in name:
                        freeze_flag = True
                if freeze_flag:
                    param.requires_grad = False
                    print(f" Frozer layer {name} ...")
                    logger.info(f" Frozer layer {name} ...")
                    
                if not freeze_flag:
                    param.requires_grad = True
            

            padding    = "max_length"
            max_length = 50
            
            dataset = load_data(dataset,test_data,"text")

            def tokenize_data(sentences):

                sentences["text"] = [
                    line for line in sentences["text"] if len(line) > 0 and not line.isspace()
                ]
                
                return tokenizer(
                    sentences["text"],
                    padding    = padding,
                    truncation = True,
                    max_length = max_length,
                    return_special_tokens_mask = True,
                )
                
            tokenized_datasets = dataset.map(
                    tokenize_data,
                    batched  = True,
                    num_proc = 10,
                    load_from_cache_file = True,
                    desc = "Running tokenizer on dataset line_by_line",
                )    

            train_dataset = tokenized_datasets["train"]
            val_dataset   = tokenized_datasets["validation"]

            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    logits = logits[0]
                return logits.argmax(dim=-1)

            metric = load_metric("accuracy")

            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm = False,
            )

            # Initialize the Trainer
            trainer = Trainer(
                model = model,
                args = training_args,
                train_dataset = train_dataset ,
                eval_dataset = val_dataset,
                tokenizer = tokenizer,
                data_collator = data_collator,
                compute_metrics = compute_metrics,
                preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            )    

            checkpoint = None
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model() 

            loss_history = {'train_loss':[], 'eval_loss':[]}
            perplexity_history = {'train_perplexity':[], 'eval_perplexity':[]}

            for log_history in trainer.state.log_history:
                if 'loss' in log_history.keys():
                    loss_history['train_loss'].append(log_history['loss'])
                    perplexity_history['train_perplexity'].append(math.exp(log_history['loss']))
                    
                elif 'eval_loss' in log_history.keys():
                    loss_history['eval_loss'].append(log_history['eval_loss'])
                    perplexity_history['eval_perplexity'].append(math.exp(log_history['eval_loss']))


            metrics = train_result.metrics

            max_train_samples = len(train_dataset)

            metrics["train_samples"] = len(train_dataset)

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate()

            max_eval_samples = len(val_dataset)
            
            metrics["eval_samples"] = len(val_dataset)
            
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            logfile_name = os.path.join(log_loc,fold_name+".json")
            print("Saving to", logfile_name)
            save_json(logfile_name, perplexity_history) 

            folders_ = os.listdir(fold_name)

            for (dirname, dirs, files) in os.walk(fold_name):
                for file in files:
                    if file.endswith('.pt'):
                        source_file = os.path.join(fold_name, file)
                        os.remove(source_file)

            
if __name__ == "__main__":
    main()


