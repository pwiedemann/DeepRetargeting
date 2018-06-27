import torch.nn as nn
import torch.nn.functional as F

'''
ENCODER and DECODER Networks 
'''

# =============================================================================
# ENCODER
# =============================================================================
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # ENCODER LAYERS
        self.conv1 = nn.Conv2d(6, 16, 15, padding=7)  
        self.conv2 = nn.Conv2d(16, 32, 15, padding=7) 
        self.conv3 = nn.Conv2d(32, 64, 15, padding=7) 
        
    def forward(self,x):
        # in: b x 6 x 256 x 256
        print("Input: ",x.shape)
        x=self.conv1(x)
        x=F.relu(x)
        x = F.dropout2d(x, p=0.2, training=self.training)
        x= F.max_pool2d(x,kernel_size=2, stride=2)  
        print("conv1: ",x.shape)
        # out: b x 16 x 128 x 128
        
        x=self.conv2(x)
        x=F.relu(x)
        x = F.dropout2d(x, p=0.2, training=self.training)
        x= F.max_pool2d(x,kernel_size=2, stride=2)  
        print("conv2: ",x.shape)
        # out: b x 32 x 64 x 64
        
        x=self.conv3(x)
        x=F.relu(x)
        x = F.dropout2d(x, p=0.2, training=self.training)
        x= F.max_pool2d(x,kernel_size=2, stride=2)  
        print("conv3: ",x.shape)
        # out: b x 64 x 32 x 32
        
        return x


# =============================================================================
# DECODER
# =============================================================================
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # DECODER LAYERS        
        self.deconv1 = nn.Conv2d(64, 32, 15, padding=7)  
        self.deconv2 = nn.Conv2d(32, 16, 15, padding=7)  
        self.deconv3 = nn.Conv2d(16, 3, 15, padding=7)  
    
    def forward(self,x):
        # in: b x 64 x 32 x 32
        print("Input: ",x.shape)
        x = F.upsample(x,scale_factor=2, mode='nearest')
        x=self.deconv1(x)
        x = F.dropout2d(x, p=0.2, training=self.training)
        x=F.relu(x)
        print("deconv1: ",x.shape)
        # out: b x 64 x 64 x 64
        
        x = F.upsample(x,scale_factor=2, mode='nearest')
        x=self.deconv2(x)
        x = F.dropout2d(x, p=0.2, training=self.training)
        x=F.relu(x)
        print("deconv2: ",x.shape)
        # out: b x 32 x 128 x 128
        
        x = F.upsample(x,scale_factor=2, mode='nearest') 
        x=self.deconv3(x)
        print("deconv3: ",x.shape)
        # out: b x 6 x 256 x 256
        return x
     


#encoder = Encoder()
#decoder = Decoder()
#
#X = torch.randn(1,6,256,256)
#
#X=encoder(X)
#X=decoder(X)
#print(str(encoder))
#print(str(decoder))


