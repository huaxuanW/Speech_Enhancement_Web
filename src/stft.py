import torch

import torch.nn as nn

import torch.nn.functional as F


def irm(clean_mag, noise_mag):
    """
    ideal ratio mask

    to recover: predicted mask * noisy mag = clean mag
    """
    eps= 1e-8
    return (clean_mag ** 2 / (clean_mag ** 2 + noise_mag ** 2 + eps)) ** 0.5

       
class torch_stft(nn.Module):
    
    def __init__(self, n_fft, hop_length, win_length, device, transform_type='logmag'):
        
        super(torch_stft, self).__init__()
        
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.win_length= win_length
        self.window = torch.hann_window(n_fft, device= device)
        self.device = device
        self.transform_type = transform_type
    
    def forward(self, dt):
        
        for key in ['mixed', 'clean', 'noise']:

            fft = torch.stft(dt[key],
                             n_fft = self.n_fft,
                             hop_length=self.hop_length,
                             win_length=self.win_length,
                             window=self.window,
                             return_complex=True).T.squeeze()
            
            mag = torch.abs(fft)
            
            if key == 'mixed':
                dt['phase'] = torch.exp(1j * torch.angle(fft))

            if self.transform_type == 'logmag':
                dt[f'{key}_mag'] = torch.log1p(mag)

        dt['mask'] = irm(dt['clean_mag'], dt['noise_mag'])
            
        return dt
    

# class torch_istft(nn.Module):
    
#     def __init__(self, n_fft, hop_length, win_length, chunk_size, device, target, transform_type, cnn):
        
#         super(torch_istft, self).__init__()
        
#         self.n_fft=n_fft
#         self.hop_length=hop_length
#         self.win_length= win_length
#         self.window = torch.hann_window(n_fft, device= 'cpu')
#         self.device = device
#         self.transform_type = transform_type
#         self.chunk_size = chunk_size
#         self.cnn = cnn
#         self.target = target
    
#     def mask_recover(self, dt):
        
#         batch, time, freq = dt['pred_mask'].shape
#         print(dt['pred_mask'].shape)
        
#         if freq == 256:
            
#             pred_mask = F.pad(dt['pred_mask'], (0, 1, 0, 0))
#             freq += 1
#             pred_mask = torch.reshape(pred_mask, (-1, freq))
        
#         else:
            
#             pred_mask = torch.reshape(dt['pred_mask'], (-1, freq))
        
#         lens = pred_mask.shape[0]
#         dt['pred_y'] = pred_mask * dt['mixed_mag'][:lens]
#         # dt['pred_y'] = torch.reshape(dt['pred_y'], (batch, time, freq))
#         return dt
   
#     def cnn2d_recover(self, dt):
        
#         # dt['pred_y'] = dt['pred_y'].reshape(-1, dt['pred_y'].shape[2])
#         lens = dt['pred_y'].shape[0]
#         dt['pred_y'] = torch.multiply(dt['pred_y'], dt['phase'][:lens])
#         dt['true_y'] = torch.multiply(dt['clean_mag'][:lens], dt['phase'][:lens])
#         dt['mixed_y'] = torch.multiply(dt['mixed_mag'][:lens], dt['phase'][:lens])
        
#         return dt
    
#     def forward(self, dt):
        
#         if self.target == 'mask':  
            
#             dt = self.mask_recover(dt)
#         else:
#             if 'pred_mask' in dt:
#                 dt['pred_y'] = dt['pred_mask']
                
#             if dt['pred_y'].shape[2] == 256:
#                 dt['pred_y'] = F.pad(dt['pred_y'], (0, 1, 0, 0))

#         if self.transform_type == 'logmag':
            
#             for key in ['pred_y']:
            
#                 dt[key] = torch.expm1(dt[key])
#                 dt[key] = torch.clamp(dt[key], min= 0)
           
#         if self.cnn == '2d':
            
#             dt = self.cnn2d_recover(dt)

#         for key in ['pred_y']:

#             dt[key] = torch.istft(dt[key].cpu().detach().T,
#                                  n_fft = self.n_fft,
#                                  hop_length=self.hop_length,
#                                  win_length=self.win_length,
#                                  window=self.window)
            
#         return dt           

    
class torch_istft(nn.Module):
    
    def __init__(self, n_fft, hop_length, win_length, chunk_size, device, target, transform_type, cnn):
        
        super(torch_istft, self).__init__()
        
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.win_length= win_length
        self.window = torch.hann_window(n_fft, device= 'cpu')
        self.device = device
        self.transform_type = transform_type
        self.chunk_size = chunk_size
        self.cnn = cnn
        self.target = target
    
    def mask_recover(self, dt):
        
        batch, time, freq = dt['pred_mask'].shape
        
        if freq == 256:
            
            pred_mask = F.pad(dt['pred_mask'], (0, 1, 0, 0))
            freq += 1
            pred_mask = torch.reshape(pred_mask, (-1, freq))
        
        else:
            
            pred_mask = torch.reshape(dt['pred_mask'], (-1, freq))
        
        dt['pred_y'] = pred_mask * dt['mixed_mag']
        dt['pred_y'] = torch.reshape(dt['pred_y'], (batch, time, freq))
        
        return dt
    
    def cnn1d_recover(self, dt):
        
        dt['pred_y'] = torch.multiply(dt['pred_y'], dt['phase'][self.chunk_size : ])
        dt['true_y'] = torch.multiply(dt['clean_mag'][self.chunk_size : ], dt['phase'][self.chunk_size : ])
        dt['mixed_y'] = torch.multiply(dt['mixed_mag'][self.chunk_size : ], dt['phase'][self.chunk_size : ])
        
        return dt
    
    def cnn2d_recover(self, dt):
        
        dt['pred_y'] = dt['pred_y'].reshape(-1, dt['pred_y'].shape[2])
        lens = dt['phase'].shape[0]
        dt['pred_y'] = torch.multiply(dt['pred_y'][:lens], dt['phase'])
        dt['true_y'] = torch.multiply(dt['clean_mag'], dt['phase'])
        dt['mixed_y'] = torch.multiply(dt['mixed_mag'][:lens], dt['phase'])
        
        return dt
    
    def forward(self, dt):
        
        if self.target == 'mask':  
            
            dt = self.mask_recover(dt)
        else:
            if 'pred_mask' in dt:
                dt['pred_y'] = dt['pred_mask']
                
            if dt['pred_y'].shape[2] == 256:
                dt['pred_y'] = F.pad(dt['pred_y'], (0, 1, 0, 0))

        if self.transform_type == 'logmag':
            
            for key in ['mixed_mag', 'clean_mag', 'pred_y']:
            
                dt[key] = torch.expm1(dt[key])
                dt[key] = torch.clamp(dt[key], min= 0)


        if self.cnn == '1d':

            dt = self.cnn1d_recover(dt)
            
        elif self.cnn == '2d':
            
            dt = self.cnn2d_recover(dt)

        for key in ['mixed_y', 'true_y', 'pred_y']:

            dt[key] = torch.istft(dt[key].cpu().detach().T,
                                 n_fft = self.n_fft,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 window=self.window)
            
        return dt  