import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable

import torch
from transformers import PreTrainedModel

def get_attention_layers(model: PreTrainedModel):
    layers = model.model.layers
    return [layer.self_attn for layer in layers]

class AttentionReweighter:
    def __init__(
        self,
        model: PreTrainedModel,
        attention_reweight
    ):
        self._model = model
        self.attention_reweight = attention_reweight
        self._hooks = []
        self.causal_mask_org = None

    def __enter__(self):
        self._register_forward_pre_hooks()
        return

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _register_forward_pre_hooks(self):
        def attn_mask_hook(layer_idx):
            def mask_attn(mod, inp):
                if layer_idx == 0:
                    self.causal_mask_org = inp[1].clone()
                        
                causal_mask = self.causal_mask_org
                attention_reweight = self.attention_reweight[:, layer_idx].unsqueeze(dim=1)

                causal_mask_size = causal_mask.shape[-2] 
                attn_mask_size = attention_reweight.shape[-2]
                target_idxs = torch.tensor(range(attn_mask_size))
                attn_source_idx = torch.tensor(range(attn_mask_size-causal_mask_size, attn_mask_size))
                causal_source_idx = torch.tensor(range(causal_mask_size))

                causal_mask[:, :, causal_source_idx[:, None], target_idxs] = attention_reweight[:,:,attn_source_idx[:, None], target_idxs]

                inp = tuple((inp[0], causal_mask))
                return inp
            return mask_attn

        for i, layer in enumerate(get_attention_layers(self._model)):
            hook = layer.register_forward_pre_hook(attn_mask_hook(i))
            self._hooks.append(hook)

def lr_policy_default(epoch, block_rate, gap):
    if block_rate>0.2:
        learning_rate = 1e-2       
    elif block_rate>0.05:
        learning_rate = 5e-3
    elif block_rate>0.02:
        learning_rate = 2e-3
    else:
        learning_rate = 1e-3
    return learning_rate

def grad_sparse_attn(model, tokenizer,
                prompts:List[str], 
                attn_mask_0: torch.Tensor = None, 
                regular = 1e5,
                max_epoch:int = 1000, 
                lr_policy:Callable = lr_policy_default,
                check_gap = 10,
                view_gap = 50):
    '''
    This function use gradient descend to find the sparse attention masks that mantain the outputs for the given prompts.
    attn_mask_0: When given, the function continues from the given attention mask.
    regular (int or float): The regularization parameter (for loss between modified probabilities and original probabilities).
    max_epoch: Rounds of GD.
    lr_policy: An adaptive learning rate function.
    '''

    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    prompt_ids = inputs['input_ids']
    batch_size = prompt_ids.shape[0]
    num_tokens = prompt_ids.shape[1]

    # Get original probabilities for later loss calculation.
    model.eval()
    outputs = model(prompt_ids)
    prob0 = outputs.logits[:, -1].softmax(dim=-1).detach().clone()
    answer_idx_target = prob0.max(dim=-1).indices

    # Initialize the attention mask.
    if attn_mask_0 == None:
        attn_mask = torch.zeros(batch_size, 32, num_tokens, num_tokens, device=model.device)
    else:
        attn_mask = attn_mask_0.to(device=model.device)

    # Set certain connection weight to -\infty, for causal connections and padding tokens.
    min_type = torch.finfo(attn_mask.dtype).min
    causal_mask = torch.triu(torch.ones_like(attn_mask), diagonal=1).bool()
    padding_mask = (1-inputs['attention_mask'])[:, None, None, :].to(bool).repeat(1, 32, num_tokens, 1)
    causal_mask[padding_mask] = 1
    attn_mask = attn_mask.masked_fill(causal_mask, min_type)

    # Setting attn_mask to be a trainable parameter.
    attn_mask = nn.Parameter(attn_mask)
    
    # For GD, we do not update connections towards bos token and connention to itself.
    update_mask = torch.tril(torch.ones_like(attn_mask), diagonal=-1).bool() # Remove self and connections under causal mask.
    update_mask[padding_mask] = 0 # Remove padding tokens.
    # Remove bos tokens. (CAN BE IMPROVED)
    update_mask[:,:,:,0] = 0
    for batch_idx in range(batch_size):
        for i in range(num_tokens):
            if padding_mask[batch_idx, 0, -1, i]:
                update_mask[batch_idx, :, :, i+1] = 0

    # We use L2 loss to measure the gap between modified probabilities and orginal probabilities.
    loss_fn = nn.MSELoss(reduction='sum')
    block_rate = 1.
    gap = 0.
    gap_ckpt = 1000.

    warning_flag = False
    epoch = 0
    epoch_ckpt = 0
    attn_mask_ckpt = attn_mask.data.detach().clone()

    model.train()
    record = []
    print(f'prob0: {prob0.max(dim=-1).values}')
    while epoch < max_epoch:
        # Apply attn_mask in the inference process.
        modified_forward = AttentionReweighter(model, attn_mask)
        with modified_forward:
            outputs = model(prompt_ids)
            prob = outputs.logits[:, -1].softmax(dim=-1)
            answer_idx = prob.max(dim=-1).indices

        # Calculate the loss.
        sparse = attn_mask.sum()
        loss_raw = 1/batch_size*loss_fn(prob, prob0)
        loss = regular*loss_raw + sparse

        # Back propagate to calculate the gradient.
        model.zero_grad()
        if attn_mask.grad is not None:
            attn_mask.grad.data.zero_()
        loss.backward()
        
        # Manually perform GD step
        with torch.no_grad():
            if not warning_flag:
                learning_rate = lr_policy(epoch, block_rate, gap)

            attn_mask[update_mask] -= learning_rate * attn_mask.grad[update_mask]
            attn_mask.clamp_(max=0)

            block_rate = torch.exp(attn_mask[update_mask]).mean().item()
            gap = loss_raw.sqrt().item()

            if block_rate<0.01:
                break

            # Print the results
            if (epoch+1)%view_gap==0:
                print(f"Epoch: {epoch}, Block rate: {block_rate:.3f}, Prob: {prob[0,:].max().item():.3f}, Target Prob: {prob0[0,:].max().item():.3f}, Gap: {gap:.3f}")
                record.append({'epoch':epoch,'attn_mask':attn_mask.detach().clone(), 'sparsity': block_rate, 'gap': gap})
            
            # Roll back when result is not good enough.
            if (epoch+1)%check_gap==0:
                if gap<2*gap_ckpt:
                    attn_mask_ckpt = attn_mask.data.detach().clone()
                    epoch_ckpt = epoch
                    gap_ckpt = gap
                    warning_flag = False

            if gap>max(gap_ckpt,0.02)*2 and epoch>50:
                learning_rate = 0.5*learning_rate
                if learning_rate<1e-6:
                    print(f'Cannot find an effective learning rate. Optimization failed.')
                    return record
                attn_mask.data = attn_mask_ckpt.detach().clone()
                epoch = epoch_ckpt
                warning_flag = True
                print(f'Falling back to previous checkpoint. New learning rate {learning_rate}')
                
            epoch += 1
    # Return a path list of attn_mask and their properties.
    return record

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