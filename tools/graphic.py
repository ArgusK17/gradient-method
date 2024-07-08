import torch
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_by_token(data, labels=None, rotation = 90, with_cbar=True, vmax=1, vmin=0, title:str=None, save_as:str=None):
    if type(data) == torch.Tensor:
        data = data.detach().clone().cpu().numpy()
    elif type(data) == np.ndarray:
        pass
    else:
        raise Warning(f'Input data must be a 2D numpy array or pytorch tensor.')

    if not len(data.shape) == 2:
        raise Warning(f'Input data with invalid dimension ({len(data.shape)}) (2 is required).')
    
    num_layers = data.shape[0]
    num_tokens = data.shape[1]

    if with_cbar:
        plt.figure(figsize=(10.5, 6))
    else:
        plt.figure(figsize=(8, 6))
    plt.imshow(data, aspect='auto', vmin=vmin, vmax=vmax)
    plt.ylim([-0.5, num_layers-0.5])
    if with_cbar:
        plt.colorbar()

    # plot tichs and lables
    plt.grid(False)
    if not labels == None:
        plt.xticks(ticks = range(num_tokens), labels = labels, rotation = rotation)
    else:
        plt.xticks(ticks = [])
    plt.ylabel('Layer')
    plt.yticks(ticks = list(range(3,num_layers,4)), labels=[str(i) for i in range(4,num_layers+1,4)])

    if not title == None:
        plt.title(title)

    if not save_as == None:
        plt.savefig(save_as)
    plt.show()

def demonstrate_by_layer(data, labels=None, rotation = 90, with_cbar=True, vmax=1, title:str=None, save_as:str=None):
    if type(data) == torch.Tensor:
        data = data.detach().clone().cpu().numpy()
    elif type(data) == np.ndarray:
        pass
    else:
        raise Warning(f'Input data must be a 2D numpy array or pytorch tensor.')
    if not len(data.shape) == 2:
        raise Warning(f'Input data with invalid dimension ({len(data.shape)}) (2 is required).')
    if data.shape[0] != data.shape[1]:
        raise Warning(f'Input data must have same size for each dimension (now {data.shape[0]} and {data.shape[1]}).')
    
    num_tokens = data.shape[0]

    if with_cbar:
        plt.figure(figsize=(8, 6.5))
    else:
        plt.figure(figsize=(6.5, 6.5))
    plt.imshow(data, aspect='auto', vmin=0, vmax=vmax)
    if with_cbar:
        plt.colorbar()

    # plot tichs and lables
    plt.grid(False)
    if not labels == None:
        plt.yticks(ticks = range(num_tokens), labels = labels, rotation = 0)
        plt.xticks(ticks = range(num_tokens), labels = labels, rotation = rotation)
    else:
        plt.yticks(ticks = [])
        plt.xticks(ticks = [])

    if not title == None:
        plt.title(title)

    if not save_as == None:
        plt.savefig(save_as)
    plt.show()