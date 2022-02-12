import torch
import torch.nn as nn

class CNN2DBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(5, 7), dilation=1, stride=(1, 2), padding=0):

        super(CNN2DBlock, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(
                channel_in, 
                channel_out, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding, 
                dilation=dilation),

            nn.BatchNorm2d(channel_out),

            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):

        return self.f(x)

class TCNN2DBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(5, 7), dilation=1, stride=(1, 2), padding=0, dropout=False, output_padding=1):

        super(TCNN2DBlock, self).__init__()

        self.f = nn.Sequential(
            nn.ConvTranspose2d(
                channel_in, 
                channel_out, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding, 
                dilation=dilation,
                output_padding=output_padding),

            nn.BatchNorm2d(channel_out),

            nn.LeakyReLU(negative_slope=0.1)
        )

        self.d = dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.f(x)
        if self.d:
            x = self.dropout(x)
        return x

class CNN2DBlockv2(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, dilation=1, stride=1, padding=0, maxpool=None):

        super(CNN2DBlockv2, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(
                channel_in, 
                channel_out, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding, 
                dilation=dilation),

            nn.BatchNorm2d(channel_out),

            nn.ReLU(),

            
        )
        self.maxpool = maxpool
        self.m = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.f(x)
        if self.maxpool:
            x = self.m(x)
        return x
class TCNN2DBlockv2(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(5, 7), dilation=1, stride=(1, 2), padding=0, output_padding=1):

        super(TCNN2DBlockv2, self).__init__()

        self.f = nn.Sequential(
            nn.ConvTranspose2d(
                channel_in, 
                channel_out, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding, 
                dilation=dilation,
                output_padding=output_padding),

            nn.BatchNorm2d(channel_out),

            nn.ReLU()
        )

        
    def forward(self, x):
        x = self.f(x)
        return x

class SENetv10(nn.Module):
    def __init__(self, ):
        super(SENetv10, self).__init__()

        e1 = CNN2DBlockv2(1, 64, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e2 = CNN2DBlockv2(64, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e3 = CNN2DBlockv2(128, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e4 = CNN2DBlockv2(128, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=True)
        e5 = CNN2DBlockv2(128, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=False)

        
        self.encoders = nn.ModuleList([e1, e2, e3, e4, e5])

        d5 = CNN2DBlockv2(128, 128, kernel_size= 3, stride= 1, padding= 1, maxpool=False)
        d4 = TCNN2DBlockv2(256, 128, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d3 = TCNN2DBlockv2(256, 128, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d2 = TCNN2DBlockv2(256, 64, kernel_size= 3, stride= 2, padding = 1, output_padding=1)
        d1 = TCNN2DBlockv2(128, 1, kernel_size= 3, stride= 2, padding = 1, output_padding=1)  
        self.decoders = nn.ModuleList([d5, d4, d3, d2, d1])

    def forward(self, dt):
        
        x = dt['x'].reshape(-1, 1, dt['x'].shape[1], dt['x'].shape[2])
        e1 = self.encoders[0](x)

        e2 = self.encoders[1](e1)

        e3 = self.encoders[2](e2)

        e4 = self.encoders[3](e3)

        e5 = self.encoders[4](e4)

        d5 = self.decoders[0](e5)

        d4 = self.decoders[1](torch.cat([d5, e4], dim=1))

        d3 = self.decoders[2](torch.cat([d4, e3], dim=1))

        d2 = self.decoders[3](torch.cat([d3, e2], dim=1))

        d1 = self.decoders[4](torch.cat([d2, e1], dim=1))

        dt['pred_mask'] = torch.squeeze(d1,1).permute(0, 2, 1)
        return dt

class SENetv11(nn.Module):
    """
    chunk_size=16 version 3 without dropout
    """
    def __init__(self, freq_bin = 257, hidden_dim = 768, num_layer = 7, kernel_size = 3):
        super(SENetv11, self).__init__()

        e1 = CNN2DBlock(1, 64, kernel_size= (5, 7), stride= (2, 1), padding= (1, 3))
        e2 = CNN2DBlock(64, 128, kernel_size= (5, 7), stride= (2, 1), padding= (2, 3))
        e3 = CNN2DBlock(128, 256, kernel_size= (5, 7), stride= (2, 1), padding= (2, 3))
        e4 = CNN2DBlock(256, 512, kernel_size= (5, 5), stride= (2, 1), padding= (2, 2))
        e5 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        e6 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        e7 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        e8 = CNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        
        self.encoders = nn.ModuleList([e1, e2, e3, e4, e5, e6, e7, e8])

        d8 = TCNN2DBlock(512, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        d7 = TCNN2DBlock(1024, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        d6 = TCNN2DBlock(1024, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        d5 = TCNN2DBlock(1024, 512, kernel_size= (5, 5), stride= (2, 2), padding = (2, 2))
        d4 = TCNN2DBlock(1024, 256, kernel_size= (5, 5), stride= (2, 1), padding = (2, 2), output_padding=(1,0))
        d3 = TCNN2DBlock(512, 128, kernel_size= (5, 7), stride= (2, 1), padding = (2, 3), output_padding=(1,0))
        d2 = TCNN2DBlock(256, 64, kernel_size= (5, 7), stride= (2, 1), padding = (2, 3), output_padding=(1,0))
        d1 = TCNN2DBlock(128, 1, kernel_size= (5, 7), stride= (2, 1), padding = (1, 3),  output_padding=(0,0))  
        self.decoders = nn.ModuleList([d8, d7, d6, d5, d4, d3, d2, d1])

    def forward(self, dt):
        x = dt['x'].reshape(-1, 1, dt['x'].shape[1], dt['x'].shape[2])
        skip_outputs = []
        for layer in self.encoders:
            x = layer(x)
            skip_outputs.append(x)
        
        skip_output = skip_outputs.pop() 
        first = True
        for layer in self.decoders:
            if first:
                first = False
            else:
                skip_output = skip_outputs.pop()
                x = torch.cat([x, skip_output], dim = 1)
            x = layer(x)
        dt['pred_mask'] = torch.squeeze(x).permute(0, 2, 1)
        return dt