import torch.nn as nn

import torch

from .stft import *

from .sliding_window import ChunkDatav2, ChunkDatav3

from .model import SENetv10, SENetv11



class SePipline(nn.Module):
    def __init__(self, version, n_fft, hop_len, win_len, window, device, chunk_size, transform_type='logmag', stft_type='torch', **kwargs):

        super(SePipline, self).__init__()
        if version == 'v11':#models_mask_limit5 # chunk_size=32 and models_mask_limit6 # chunk_size=128 and # models_mask_limit7 #chunk_size = 64 and #models_mask_limit8 # chunk_size=16 #models_mask_limit9 chunk_size=128 with snr_mixer2
            print(version)
            chunk = ChunkDatav3(chunk_size= chunk_size, target= 'mask')
            model = SENetv10()
        elif version == 'v12':
            print(version) #models_mask2
            chunk = ChunkDatav2(chunk_size= 16, target= 'mask')
            model = SENetv11()

        
        if stft_type == 'torch':
            _stft = torch_stft(n_fft=n_fft, hop_length=hop_len, win_length= win_len, device = device, transform_type= transform_type)
            _istft = torch_istft(n_fft =n_fft, hop_length=hop_len, win_length= win_len, device=device, chunk_size= chunk_size, transform_type =transform_type, target= kwargs['target'], cnn = kwargs['cnn'])
            

    #     self.model = nn.Sequential(
    #         _stft,
    #         chunk,
    #         model,
    #         _istft
    #     ).to(device)

    # def forward(self, dt):

    #     dt = self.model(dt)

    #     return dt

    
        self.stft = _stft.to(device)
        self.chunk = chunk.to(device)
        self.model = model.to(device)
        self.istft = _istft.to(device)

    def forward(self, dt, train=True):

        dt = self.stft(dt)
        dt = self.chunk(dt)
        dt = self.model(dt)
        if not train:
            dt = self.istft(dt)

        return dt

def load_model(model, optimizer, action='train', **kargs):
    
    if action == 'train':
        
        print('train from begin')
    
        epoch = 1
        
        return epoch, model, optimizer
    
    elif action == 'retrain':
        
        print(f"load model from {kargs['pretrained_model_path']}")
        
        checkpoint = torch.load(kargs['pretrained_model_path'])
        
        epoch = checkpoint['epoch'] + 1
        
        model.eval()
        
        model.load_state_dict(checkpoint['model'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        for state in optimizer.state.values():
            
            for k, v in state.items():
                
                if isinstance(v, torch.Tensor):
                    
                    state[k] = v.cuda()
                    
        return epoch, model, optimizer
    
    elif action == 'predict':
        
        print(f"load model from {kargs['pretrained_model_path']}")
        
        checkpoint = torch.load(kargs['pretrained_model_path'], map_location= 'cpu')  
        
        model.eval()
        
        model.load_state_dict(checkpoint['model'])
        
        return model