import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def forward(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        prompts,
        with_hiddens: bool = False, 
        with_attns: bool = False, 
        gen_len: int = 1
        ):
    
    '''
    Standard batch generation.
    '''

    if type(prompts) == list:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    elif type(prompts) == torch.Tensor:
        inputs = {'input_ids': prompts.to(model.device), 'attention_mask':torch.ones_like(prompts).to(model.device)}
    
    outputs = model.generate(
        **inputs,
        do_sample = False,
        return_dict_in_generate = True,
        output_scores=True,
        output_hidden_states = with_hiddens,
        output_attentions = with_attns,
        max_new_tokens = gen_len,
        pad_token_id = tokenizer.eos_token_id
    )
    return outputs

def get_attention_layers(model: PreTrainedModel):
    layers = model.model.layers
    return [layer.self_attn for layer in layers]

class AttentionInterceptor:
    def __init__(
        self,
        model: PreTrainedModel,
        attention_mask: torch.Tensor
    ):
        self._model = model
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(bool)
        self.attention_mask = attention_mask
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
                attention_mask = self.attention_mask[:, layer_idx].unsqueeze(dim=1) # Take specific layer, but unsqueeze to fit different heads.

                mask_size = min(causal_mask.shape[-1], attention_mask.shape[-1])
                target_idxs = torch.tensor(range(mask_size))
                source_idx = torch.tensor(range(mask_size))

                min_dtype = torch.finfo(causal_mask.dtype).min
                padding_mask = causal_mask[:, :, source_idx[:, None], target_idxs].eq(0.0) * attention_mask[:,:,source_idx[:, None], target_idxs]
                if padding_mask.any():
                    causal_mask[:, :, source_idx[:, None], target_idxs] = causal_mask[:, :, source_idx[:, None], target_idxs].masked_fill(padding_mask, min_dtype)

                inp = tuple((inp[0], causal_mask))
                return inp
            return mask_attn

        for i, layer in enumerate(get_attention_layers(self._model)):
            hook = layer.register_forward_pre_hook(attn_mask_hook(i))
            self._hooks.append(hook)

class AttentionInterceptor_Mistral(AttentionInterceptor):
    def __init__(
        self,
        model: PreTrainedModel,
        attention_mask: torch.Tensor
    ):
        self._model = model
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(bool)
        self.attention_mask = attention_mask
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
                # print(causal_mask.shape)
                attention_mask = self.attention_mask[:, layer_idx].unsqueeze(dim=1) # Take specific layer, but unsqueeze to fit different heads.

                causal_mask_size = causal_mask.shape[-2] 
                attn_mask_size = attention_mask.shape[-2]
                target_idxs = torch.tensor(range(attn_mask_size))
                attn_source_idx = torch.tensor(range(attn_mask_size-causal_mask_size, attn_mask_size))
                causal_source_idx = torch.tensor(range(causal_mask_size))

                min_dtype = torch.finfo(causal_mask.dtype).min
                padding_mask = causal_mask[:, :, causal_source_idx[:, None], target_idxs].eq(0.0) * attention_mask[:,:,attn_source_idx[:, None], target_idxs]
                if padding_mask.any():
                    causal_mask[:, :, causal_source_idx[:, None], target_idxs] = causal_mask[:, :, causal_source_idx[:, None], target_idxs].masked_fill(padding_mask, min_dtype)

                inp = tuple((inp[0], causal_mask))
                return inp
            return mask_attn

        for i, layer in enumerate(get_attention_layers(self._model)):
            hook = layer.register_forward_pre_hook(attn_mask_hook(i))
            self._hooks.append(hook)