import torch
import numpy as np

# Model Loading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from tools.transformers_patch import forward_mod

import json
import random
import glob

def gen_prompt(num_examples=3, seed=0, task='en2fr', sep_token='\n'):    
    random.seed(seed)
    packed_prompts={}

    pattern = './datasets/ICL/*.json'
    fname_list = glob.glob(pattern)
    task_list = [fname.split('/')[-1].split('.')[0] for fname in fname_list]
    if not task in task_list:
        raise Warning(f'Invalid task \'{task}\'. Please select from {task_list}')

    with open(f'./datasets/ICL/{task}.json', mode='r', encoding='utf-8') as file:
        data = json.load(file)

    data_len = data['metadata']['num_examples']
    idx_list = random.sample(range(data_len), num_examples+1)

    prompt=""
    for idx in idx_list[:-1]:
        prompt=prompt+data['examples'][str(idx+1)]['input']+"->"+data['examples'][str(idx+1)]['output']+sep_token
    prompt=prompt+data['examples'][str(idx_list[-1]+1)]['input']+"->"
    
    packed_prompts['icl'] = prompt
    packed_prompts['ins'] = data['instruct']+data['examples'][str(idx_list[-1]+1)]['input']+"->"
    packed_prompts['ans'] = data['examples'][str(idx_list[-1]+1)]['output']

    return packed_prompts

import csv
def gen_prompt_comp(idx, model_type = 'llama'):

    file_path = './datasets/GSM8K/main_train.csv'
    dataset = []
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            dataset.append(row)
    packed_prompts = {}
    packed_prompts['pre'] = f"[Question] {dataset[idx+1][0]} \n[Answer] Let\'s think step by step."
    if model_type =='llama' or model_type =='mistral':
        packed_prompts['ins'] = f"[INST] {dataset[idx+1][0]} [/INST] "
    elif model_type == 'gemma':
        packed_prompts['ins'] = f"<start_of_turn>user\n{dataset[idx+1][0]}<end_of_turn>\n<start_of_turn>model\n"

    packed_prompts['answer'] = dataset[idx+1][1]

    return packed_prompts

def seed_all(seed):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(path, model_type='llama'):
    if model_type == 'llama':
        LlamaDecoderLayer.forward = forward_mod
    elif model_type == 'mistral':
        MistralDecoderLayer.forward = forward_mod
    elif model_type == 'gemma':
        GemmaDecoderLayer.forward = forward_mod
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, device_map='cuda', attn_implementation='eager') # Use load_in_8bit=True if CUDA out of memory.

    return tokenizer, model

def show_outputs(tokenizer, output_ids, without_prompts=None, in_tokens=False):
    if without_prompts==None:
        start_pos = 0
    else:
        if type(without_prompts) == list:
            inputs = tokenizer(without_prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            start_pos = input_ids.shape[1]
        if type(without_prompts) == torch.Tensor:
            start_pos = without_prompts.shape[1]

    responses=[]
    for s in output_ids[:, start_pos:]:
        if not in_tokens:
            response = tokenizer.decode(s, skip_special_tokens=True)
        else:
            response = []
            for t in s:
                response.append(tokenizer.decode(t, skip_special_tokens=True))
        responses.append(response)
    return responses

def group_tokens(tokenizer, sequences):
    batch_size = sequences.shape[0]
    num_tokens = sequences.shape[1]

    target_tokens = {'<s>':0, '->':0, '\n':0}
    target_inputs = tokenizer(['a->b\nc->d\ne->'], return_tensors='pt')
    target_tokens['<s>'] = target_inputs['input_ids'][0, 0].item()
    target_tokens['->'] = target_inputs['input_ids'][0, 2].item()
    target_tokens['\n'] = target_inputs['input_ids'][0, 4].item()

    num_examples = torch.where(sequences[0]==target_tokens['\n'])[0].shape[0]
    token_index_group = np.zeros((batch_size, 4*num_examples+2), dtype=object)

    for batch_idx in range(batch_size):
        start_flag = False
        group = []
        group_idx = 0
        for token_idx in range(num_tokens):
            token = sequences[batch_idx, token_idx].item()
            if start_flag:
                if token == target_tokens['->'] or token == target_tokens['\n']:
                    token_index_group[batch_idx, group_idx] = torch.tensor(group)
                    group_idx+=1
                    token_index_group[batch_idx, group_idx] = torch.tensor([token_idx])
                    group_idx+=1
                    group = []
                else:
                    group.append(token_idx)
            if token == target_tokens['<s>']:
                start_flag = True
    return token_index_group

def reshape(response:str):
    response_re=response
    while response_re!="" and (response_re[0]==' ' or response_re[0] == '\n'):
        response_re=response_re[1:]
    response_re=response_re.split('\n')[0]
    response_re=response_re.split('->')[0]
    response_re=response_re.split(' ')[0]
    return response_re

def is_correct(response, answer, to_chr=-1):
    res_len = len(reshape(response))
    ans_len = len(reshape(answer))
    if to_chr>0:
        mod_len=min(to_chr,res_len)
    else:
        mod_len=res_len

    if mod_len>0:
        response_mod=reshape(response)[:min(mod_len,ans_len)]
        answer_mod=reshape(answer)[:min(mod_len,ans_len)]
    else:
        response_mod=""
        answer_mod=answer

    return response_mod==answer_mod
