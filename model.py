import torch
import torch.nn as nn

class Generator(nn.Module):
    """Generator network for sketch completion"""
    def __init__(self, input_channels=1, output_channels=1):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(input_channels, 64, normalize=False)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 512)
        
        # Decoder
        self.dec4 = self.deconv_block(512, 512)
        self.dec3 = self.deconv_block(1024, 256)
        self.dec2 = self.deconv_block(512, 128)
        self.dec1 = self.deconv_block(256, 64)
        
        # Output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)
    
    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections (U-Net style)
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        return self.final(d1)


class Discriminator(nn.Module):
    """Discriminator network to distinguish real from generated sketches"""
    def __init__(self, input_channels=2):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: (input + target) concatenated
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)


def initialize_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
