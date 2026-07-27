import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm

from model import Generator, Discriminator, initialize_weights
from feature_extractor import SketchFeatureExtractor

class SketchDataset(Dataset):
    """Dataset for sketch completion"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.extractor = SketchFeatureExtractor()
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        
        # Full sketch (target)
        target = self.extractor.normalize(image)
        
        # Create partial sketch (input) by masking random regions
        input_sketch = target.copy()
        mask_size = np.random.randint(30, 100)
        x, y = np.random.randint(0, 256-mask_size, 2)
        input_sketch[y:y+mask_size, x:x+mask_size] = -1.0
        
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_sketch).unsqueeze(0)
        target_tensor = torch.FloatTensor(target).unsqueeze(0)
        
        return input_tensor, target_tensor


def train(args):
    """Train the GAN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    initialize_weights(generator)
    initialize_weights(discriminator)
    
    # Loss functions
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Dataset
    if os.path.exists(args.data_dir):
        dataset = SketchDataset(args.data_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        print(f"Warning: Data directory '{args.data_dir}' not found. Creating dummy data...")
        os.makedirs(args.data_dir, exist_ok=True)
        print("Please add sketch images to the 'data/sketches' directory and restart training.")
        return
    
    # Training loop
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (input_sketches, target_sketches) in enumerate(pbar):
            input_sketches = input_sketches.to(device)
            target_sketches = target_sketches.to(device)
            
            batch_size = input_sketches.size(0)
            real_label = torch.ones(batch_size, 1, 30, 30).to(device)
            fake_label = torch.zeros(batch_size, 1, 30, 30).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            fake_sketches = generator(input_sketches)
            
            pred_real = discriminator(input_sketches, target_sketches)
            loss_D_real = criterion_GAN(pred_real, real_label)
            
            pred_fake = discriminator(input_sketches, fake_sketches.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake_label)
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            pred_fake = discriminator(input_sketches, fake_sketches)
            loss_G_GAN = criterion_GAN(pred_fake, real_label)
            loss_G_L1 = criterion_L1(fake_sketches, target_sketches) * args.lambda_l1
            
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            
            pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}'
            })
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, f'checkpoints/model_epoch_{epoch+1}.pth')
            print(f"\nCheckpoint saved at epoch {epoch+1}")
    
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train sketch completion GAN')
    parser.add_argument('--data_dir', type=str, default='data/sketches', help='Path to sketch dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='L1 loss weight')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    train(args)
