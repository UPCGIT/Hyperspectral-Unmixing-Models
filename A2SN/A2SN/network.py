import torch
import torch.nn as nn
import torch.nn.functional as F


class Spectral(nn.Module):
    def __init__(self,L):
        super(Spectral, self).__init__()
        self.conv11 = nn.Sequential(
        nn.Conv2d(L, 8, kernel_size=1, stride=1, padding=0),
        )
        
        self.batch_norm11= nn.Sequential(
        nn.BatchNorm2d(8),
        nn.ReLU(),
        )
        
        self.conv12 = nn.Sequential(
        nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
        )
        
        self.batch_norm12= nn.Sequential(
        nn.BatchNorm2d(16),
        nn.ReLU(),
        )
        
        self.conv13 = nn.Sequential(
        nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0),
        )
        
        self.batch_norm13= nn.Sequential(
        nn.BatchNorm2d(24),
        nn.ReLU(),
        )
        
        self.conv14 = nn.Sequential(
        nn.Conv2d(24, 8, kernel_size=1, stride=1, padding=0),
        )
        
        self.batch_norm14= nn.Sequential(
        nn.BatchNorm2d(32),
        nn.ReLU(),
        )
        
        self.conv15 = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x11 = self.conv11(x)
        
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x11)
        
        x13 = torch.cat((x11,x12),dim=1)        
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        
        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)
        
        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x15 = self.batch_norm14(x15)
        x16 = self.conv15(x15)
        return x16


class Spatial(nn.Module):
    def __init__(self,L):
        super(Spatial, self).__init__()
        self.conv11 = nn.Sequential(
        nn.Conv2d(L, 8, kernel_size=3, stride=1, padding=1),
        )
        
        self.batch_norm11= nn.Sequential(
        nn.BatchNorm2d(8),
        nn.ReLU(),
        )
        
        self.conv12 = nn.Sequential(
        nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
        )
        
        self.batch_norm12= nn.Sequential(
        nn.BatchNorm2d(16),
        nn.ReLU(),
        )
        
        self.conv13 = nn.Sequential(
        nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
        )
        
        self.batch_norm13= nn.Sequential(
        nn.BatchNorm2d(24),
        nn.ReLU(),
        )
        
        self.conv14 = nn.Sequential(
        nn.Conv2d(24, 8, kernel_size=3, stride=1, padding=1),
        )
        
        self.batch_norm14= nn.Sequential(
        nn.BatchNorm2d(32),
        nn.ReLU(),
        )
        
        self.conv15 = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x11 = self.conv11(x)
        
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x11)
        
        x13 = torch.cat((x11,x12),dim=1)        
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        
        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)
        
        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x15 = self.batch_norm14(x15)
        x16 = self.conv15(x15)
        return x16


class EE_Abu(nn.Module):
    def __init__(self, B, L, P):
        super(EE_Abu, self).__init__()
                                                  
        self.convm0 = nn.Sequential(
        nn.Conv2d(B, L, kernel_size=1, padding=0),
        nn.Sigmoid(),
        )
        self.convm1 = nn.Sequential(
        nn.Conv2d(L, P, kernel_size=1, padding=0),
        nn.BatchNorm2d(P),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, identity=None):
        
        if identity is None:
            identity = x  # NCHW
        n,c,h,w = identity.size()
        x = self.convm0(x) # NCHW ==> NDHW
        n,b,h,w = x.size()
        
        abu = self.convm1(x)
        abu = self.softmax(abu)
        n,d,h,w = abu.size()
        
        x = x.view(n,b,-1)
        abu = abu.view(n,d,-1)
        abu = abu.permute(0,2,1)
        
        end = torch.matmul(x,abu)
        end = F.normalize(end,p=2,dim=1)

        abu = abu.permute(0,2,1)
        re = torch.matmul(end,abu)

        abu = abu.squeeze(0)
        end = end.squeeze(0)
        re = re.squeeze(0)
        xo = x.squeeze(0)
        return abu, end, re, xo  


class A2SN(nn.Module):
    def __init__(self,L, P):
        super(A2SN, self).__init__()
        self.spe = Spectral(L)
        self.spa = Spatial(L)
        self.result = EE_Abu(64, L, P)
   
    def forward(self, x):
        x1 = self.spe(x)
        x2 = self.spa(x)
        x3 = torch.cat((x1,x2),dim=1) 
        abu, end, re, xo = self.result(x3)
        return abu, end, re, xo
    
