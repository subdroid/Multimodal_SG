import argparse
import logging

import numpy as np
import torch

def generate(model,tokenizer,sent,correct_token_id,temperature):

    # print(sent)
    
    device          =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    
    encoded_prompt  =  tokenizer.encode(sent, add_special_tokens=False, return_tensors="pt")
    encoded_prompt  =  encoded_prompt.to(device)
    # print(encoded_prompt)
    
    output_sequences = model.generate(
        do_sample = True,
        input_ids=encoded_prompt,
        max_length=len(encoded_prompt[0])+1,
        temperature=temperature,
        output_scores=True,
        return_dict_in_generate=True,
    )
    generated_words = output_sequences.sequences[:,encoded_prompt.shape[-1]:] # -> shape [1, 5]
    
    probs = torch.stack(output_sequences.scores, dim=1).softmax(-1)  # -> shape [1, 5, vocab_size]
    

    prob_first_token = probs[0][0]
    
    predicted_token_id  =  torch.argmax(prob_first_token)
    predicted_token     =  tokenizer.decode(predicted_token_id)
    correct_token       =  tokenizer.decode(correct_token_id)
    
    
    first_token_softmax_list = prob_first_token.tolist()
    confidence =  (prob_first_token.tolist())[predicted_token_id]
    accuracy =  (prob_first_token.tolist())[correct_token_id]
    
    
    
    return predicted_token,correct_token,confidence,accuracy
    