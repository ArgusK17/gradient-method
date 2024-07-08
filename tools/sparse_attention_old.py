import torch
from collections import deque
from typing import Dict, List
import random
from tqdm.notebook import tqdm
from IPython.display import display, clear_output

from tools.inference import AttentionInterceptor, AttentionInterceptor_Mistral

def propose_update(attn_mask, search_mask, num_changes: torch.Tensor, p: torch.Tensor, mode = 'token'):
    attn_mask_mod = attn_mask.clone()

    batch_size = attn_mask.shape[0]
    update_result = torch.zeros(batch_size, dtype = bool)

    for idx in range(batch_size):
        num_changes_loc = num_changes[idx].item()
        p_loc = p[idx].item()
        # Change num_changes_loc times. (may repeat)
        for _ in range(num_changes_loc):
            r = torch.rand(1).item()
            if r<p_loc:
                target = False
            else:
                target = True

            blocking_cand = torch.where((attn_mask[idx] == target) & search_mask[idx])
            if len(blocking_cand[0])>0:
                index_idx = torch.randint(0, len(blocking_cand[0]), (1,)).item()
                
                source_idx = blocking_cand[-2][index_idx].item()
                target_idx = blocking_cand[-1][index_idx].item()
                if mode == 'token':
                    attn_mask_mod[idx, :, source_idx:, target_idx] = (not target)
                elif mode == 'layer':
                    layer_idx = blocking_cand[0][index_idx].item()
                    attn_mask_mod[idx, layer_idx, source_idx:, target_idx] = (not target)
                update_result[idx] = True

    return attn_mask_mod, update_result

def eval_attn_mask(model, input_ids, output_ids_real, attention_mask, start_pos, 
                   fast_infer = False, past_key_values=None):

    mask_size = attention_mask.shape[-1]
    num_tokens = input_ids.shape[-1]

    if fast_infer and past_key_values == None:
        raise Warning(f'past_key_values must be provided for fast inference.')

    attn_mask = attention_mask.clone().to(model.device)
    
    if mask_size != num_tokens:
        raise Warning(f'Attention mask size ({mask_size}) mismatch num_tokens ({num_tokens})')
    else:
        architecture = model.config.architectures[0]
        if architecture == 'MistralForCausalLM':
            forward_modifier = AttentionInterceptor_Mistral(model, attention_mask = attn_mask)
        elif architecture == 'LlamaForCausalLM' or architecture == 'GemmaForCausalLM':
            forward_modifier = AttentionInterceptor(model, attention_mask = attn_mask)
        else:
            print(f'Unsupported Architecture: {architecture}. Treat as default (Llama). May cause cuda error.')
            forward_modifier = AttentionInterceptor(model, attention_mask = attn_mask)
        with forward_modifier:
            if fast_infer:
                history_num_tokens = past_key_values[0][0].shape[-2]
                outputs = model(input_ids[:, history_num_tokens:], past_key_values = past_key_values)
                result_ids = outputs.logits.max(dim=-1).indices
            else:
                outputs = model(input_ids)
                result_ids = outputs.logits.max(dim=-1).indices[:, start_pos:]

    evaluation = (result_ids == output_ids_real).all(dim=-1)
    return evaluation

def get_past_key_values(outputs0, batch_size, start_pos):
    past_key_values_list = [[None, None] for _ in outputs0.past_key_values]
    for idx, pkv in enumerate(outputs0.past_key_values):
        past_key_values_list[idx][0] = pkv[0][:,:,:start_pos].repeat(batch_size,1,1,1)
        past_key_values_list[idx][1] = pkv[1][:,:,:start_pos].repeat(batch_size,1,1,1)
    past_key_values = tuple([tuple(pkv) for pkv in past_key_values_list])
    return past_key_values

def select_pkv(past_key_values: tuple, optim_flag: torch.tensor):
    past_key_values_list = [[None, None] for _ in past_key_values]
    for pkv_idx, pkv in enumerate(past_key_values):
        past_key_values_list[pkv_idx][0] = pkv[0][optim_flag]
        past_key_values_list[pkv_idx][1] = pkv[1][optim_flag]
    past_key_values_selected = tuple([tuple(pkv) for pkv in past_key_values_list])
    return past_key_values_selected

def attn_mask_sparse(model, outputs0,
                     batch_size, start_pos, end_pos=-1,
                     attn_mask_0 = None,
                     **kwargs: Dict,
                     ):
    # Use a dict optim_config to set num_rounds, num_demo_rounds, mode and other useful configs
    '''
    Return the most sparse attn_mask possible (given by random search) for generating the same outputs.
    For gen_diff = False, it means that all generation use the same attention_mask, i.e. ignore the same tokens at same spot
    '''

    config  = {
        'mode' : 'token',
        'num_rounds' : 1000,
        'silent' : True,
        'fast_infer' : False,
        'num_demo_rounds' : 50,
        'record_window' : 50,
        'p_bot': 0.8,
        'search_range' : list(range(1, start_pos)),
    }
    config.update(kwargs)

    num_layers = model.config.num_hidden_layers
    num_tokens = outputs0.sequences.shape[1]

    if end_pos<0:
        end_pos = num_tokens + end_pos # config['end_pos'] = -1 -> end_pos = num_tokens-1

    # Attention mask intialization
    if attn_mask_0 == None:
        attn_mask = torch.zeros(batch_size, num_layers, end_pos, end_pos, dtype = bool).to(model.device)
    else:
        attn_mask = attn_mask_0.clone().to(model.device)

    input_ids = outputs0.sequences[:, :end_pos].repeat(batch_size, 1)
    if config['fast_infer']:
        past_key_values = get_past_key_values(outputs0, batch_size, start_pos)
    else:        
        attn_mask[:, :, :start_pos] = 0
        past_key_values = None
    output_ids_real = outputs0.sequences[:,start_pos+1:end_pos+1].repeat(batch_size,1)

    # Create search_mask to represent legal search position
    search_mask = torch.zeros_like(attn_mask).to(model.device)
    if config['mode'] == 'token':
        search_mask[:, 0, start_pos, config['search_range']] = 1
    elif config['mode'] == 'layer':
        search_mask = (~attn_mask.all(dim=1)).unsqueeze(dim=1).repeat(1, num_layers, 1, 1)
        search_mask[:, :, :start_pos, :] = 0
        if search_mask.shape[-2]>start_pos+1:
            search_mask[:, :, start_pos+1:, :] = 0
        search_mask[:, :, start_pos, 0] = 0
        search_mask[:, :, start_pos, start_pos:] = 0

        for pos in range(start_pos):
            if not (pos in config['search_range']):
                search_mask[:, :, start_pos, pos] = 0

    accept_rate_record = deque(maxlen=config['record_window'])
    block_rate = 0
    optim_flag = torch.ones(batch_size, dtype = bool).to(model.device)

    # Check whether the initial attention mask is correct.
    evaluation = eval_attn_mask(model, input_ids, output_ids_real, attn_mask, start_pos, 
                                                fast_infer=config['fast_infer'], past_key_values=past_key_values)
    if not evaluation.all():
        print('Initialization failed due to numerical error. Please use float32 or change your prompts.')
        return None

    num_changes = torch.ones(batch_size, dtype = int) 
    for round_idx in range(config['num_rounds']):
        if round_idx < 10:
            p = 1*torch.ones(batch_size)
        elif round_idx< int(config['num_rounds']/2):
            p = config['p_bot']*torch.ones(batch_size)
        else:
            p = torch.ones(batch_size)

        # Propose attn_mask_mod
        attn_mask_mod, update_result = propose_update(attn_mask, search_mask, num_changes, p, mode = config['mode'])

        # Evaluate attn_mask_mod
        evaluation = torch.zeros(batch_size, dtype = bool).to(model.device)

        if config['fast_infer']:
            past_key_values_select = select_pkv(past_key_values, optim_flag)
        else:
            past_key_values_select = None
        evaluation[optim_flag] = eval_attn_mask(model, input_ids[optim_flag], output_ids_real[optim_flag], attn_mask_mod[optim_flag], start_pos, 
                                                fast_infer=config['fast_infer'], past_key_values=past_key_values_select)
        accept = torch.zeros(batch_size)
        for idx in torch.where(optim_flag)[0]:
            if update_result[idx] and evaluation[idx]:
                attn_mask[idx] = attn_mask_mod[idx].clone()
                accept[idx] = 1

        # Calculate accept and block rate
        accept_rate_record.append(accept)
        accept_rate = torch.stack(tuple(accept_rate_record), dim = 0).mean(dim=0).to(model.device)
        block_rate = attn_mask[:, :, start_pos, config['search_range']].to(float).mean(dim = -1).mean(dim = 1)

        for idx in torch.where(optim_flag)[0]:
            if accept_rate[idx] < 0.2 and num_changes[idx]>1:
                num_changes[idx] = max(int(num_changes[idx]/2),1)
            elif accept_rate[idx] > 0.4 and num_changes[idx]<10:
                num_changes[idx] = num_changes[idx]*2

        if config['mode'] == 'token':
            optim_flag = (block_rate <= (1-10/num_tokens)) & ((accept_rate > 0) | (round_idx < config['num_rounds']/2))
        elif config['mode'] == 'layer':
            optim_flag = (accept_rate > 0) | (round_idx < config['num_rounds']/2)

        if not optim_flag.any():      
            break

        if not config['silent'] and (round_idx+1)%config['num_demo_rounds'] == 0:
            # clear_output(wait=True)
            # print('Optimizing...')
            print(f"Round {round_idx+1:02d}, Accept Rate: {accept_rate.mean().item()*100:.2f}%, "
                +f"Block Rate: {block_rate.mean().item()*100:.2f}%")
            print(f'Number of changes: {num_changes.to(float).mean():.2f}, Batch optimizing: {optim_flag.to(int).sum()}')

    if not config['silent']:
        print(f"{config['mode']} optimization complete! Block Rate: {block_rate.mean().item()*100:.2f}%")

    return attn_mask

def attn_mask_clean_up(model, outputs0,
                     batch_size, start_pos, end_pos = -1,
                     attn_mask_0 = None,
                     **kwargs: Dict,
                     ):
    
    if attn_mask_0==None:
        return None

    attn_mask = attn_mask_0.clone()
    config  = {
        'mode' : 'token',
        'num_rounds' : 1000,
        'silent' : True,
        'fast_infer' : False,
        'num_demo_rounds' : 50,
        'record_window' : 20,
        'search_range' : range(1, start_pos),
    }
    config.update(kwargs)
    num_layers = model.config.num_hidden_layers
    num_tokens = outputs0.sequences.shape[1]

    if end_pos<0:
        end_pos = num_tokens + end_pos # config['end_pos'] = -1 -> end_pos = num_tokens

    input_ids = outputs0.sequences[:, :end_pos].repeat(batch_size, 1)
    if config['fast_infer']:
        past_key_values = get_past_key_values(outputs0, batch_size, start_pos)
    else:        
        attn_mask[:, :, :start_pos] = 0
        past_key_values = None
    output_ids_real = outputs0.sequences[:,start_pos+1:end_pos+1].repeat(batch_size,1)

    block_rate = attn_mask[:, :, start_pos, config['search_range']].to(float).mean(dim = -1).mean(dim = 1)
    if not config['silent']:
        print(f"Cleaning begin. Block Rate: {block_rate.mean().item()*100:.2f}%")

    # layer_list = list(range(num_layers))
    # token_list = list(range(1, start_pos))
    # random.shuffle(layer_list)
    # if not config['silent']:
    #     layer_list_demo = tqdm(layer_list)
    # else:
    #     layer_list_demo = layer_list

    random_list = []
    for layer_idx in range(num_layers):
        for token_idx in config['search_range']:
            random_list.append((layer_idx, token_idx,))
    random.shuffle(random_list)

    for epoch, (layer_idx, token_idx) in enumerate(random_list):
        # Propose attn_mask_mod
        attn_mask_mod = attn_mask.clone()
        optim_flag = (attn_mask[:, layer_idx, start_pos, token_idx] == 0)
        attn_mask_mod[:, layer_idx, start_pos:, token_idx] = 1

        # Evaluate attn_mask_mod
        if optim_flag.any():
            evaluation = torch.zeros(batch_size, dtype = bool).to(model.device)
            if config['fast_infer']:
                past_key_values_select = select_pkv(past_key_values, optim_flag)
            else:
                past_key_values_select = None
            evaluation[optim_flag] = eval_attn_mask(model, input_ids[optim_flag], output_ids_real[optim_flag], attn_mask_mod[optim_flag], start_pos, 
                                                    fast_infer=config['fast_infer'], past_key_values=past_key_values_select)
            for idx in torch.where(optim_flag)[0]:
                if evaluation[idx]:
                    attn_mask[idx] = attn_mask_mod[idx].clone()

        if not config['silent'] and (epoch+1)%(start_pos-1)==0:
            # clear_output(wait=True)
            # print('Cleaning the result.')
            block_rate = attn_mask[:, :, start_pos, config['search_range']].to(float).mean(dim = -1).mean(dim = 1)
            print(f'{epoch+1}/{len(random_list)} of cleaning done. Block Rate: {block_rate.mean().item()*100:.2f}%')
           
    block_rate = attn_mask[:, :, start_pos, config['search_range']].to(float).mean(dim = -1).mean(dim = 1)
    if not config['silent']:
        # clear_output(wait=True)
        print(f'Cleaning done. Block Rate: {block_rate.mean().item()*100:.2f}%')

    return attn_mask

from tools.inference import forward
from tools.utils import show_outputs, is_correct
import time

def auto_highlight_icl(tokenizer, sequence):
    idx_list = []

    inputs = tokenizer(['a->b\nc->d\ne->'], return_tensors='pt')
    target_tokens = inputs['input_ids'][0, 4].to('cuda')

    for idx, token in enumerate(sequence):
        if token == target_tokens:
            idx_list.append(idx)
    return idx_list

def search_attn_mask(tokenizer, model, prompt, answer, gen_len=3, batch_size=20, silent=True):

    prompt_ids = tokenizer([prompt], return_tensors="pt", padding=True)['input_ids']
    start_pos = prompt_ids.shape[1]-1

    outputs0 = forward(model, tokenizer, prompt_ids, gen_len=gen_len)

    gen_len_list = auto_highlight_icl(tokenizer, outputs0.sequences[0, start_pos+1:])
    if len(gen_len_list)>0:
        end_pos = start_pos+gen_len_list[0]
    else:
        end_pos = outputs0.sequences.shape[-1]-1

    results0 = show_outputs(tokenizer, outputs0.sequences[:, :end_pos+1], without_prompts=[prompt])
    
    print(results0, answer)
    
    if answer == None or is_correct(results0[0], answer):
        start_time=time.time()
        search_range = range(1, start_pos)
        attn_mask = attn_mask_sparse(model, outputs0, batch_size, start_pos, end_pos, 
                                    fast_infer=True, mode='token', num_rounds=200, 
                                    search_range = search_range, silent=silent)
        attn_mask = attn_mask_sparse(model, outputs0, batch_size, start_pos, end_pos, attn_mask_0=attn_mask, 
                                    fast_infer=True, mode='layer', num_rounds=1000, 
                                    search_range = search_range, silent=silent)
        attn_mask = attn_mask_clean_up(model, outputs0, batch_size, start_pos, end_pos, attn_mask_0=attn_mask, 
                                    fast_infer=True, search_range=search_range, silent=silent)
            
        return outputs0.sequences[:, :end_pos+1], attn_mask[:,:,start_pos,:start_pos+1]
    else:
        if not silent:
            print(f'LLM cannot handle this prompt. Please change you prompt.')
        return outputs0.sequences[:, :end_pos+1], None
