import torch
import torch.nn as nn
import math

class crm(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=False):
        super().__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels
            if self.inter_channels == 0:
                self.inter_channels = 1

        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
       
        bn = nn.BatchNorm2d
        self.softmax = nn.Softmax(dim=-1)

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self,x,y):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # b x hw x c
        
        theta_x = self.theta(y).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # b x hw x c 
            
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # b x c x hw
        f = torch.matmul(theta_x, phi_x)  # b x hw x hw     
        N = f.size(-1)
        f_div_C = f / N  
        #f_div_C=self.softmax(f)
        y = torch.matmul(f_div_C, g_x)  # b x hw x c
        y = y.permute(0, 2, 1).contiguous()  # b x c x hw
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # b x c x h x w  
        W_y = self.W(y)     
        z = W_y + x
        return z

