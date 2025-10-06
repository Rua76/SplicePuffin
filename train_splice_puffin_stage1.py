import os
import sys
import time
import h5py
import argparse
import pandas as pd
import numpy as np
import pyBigWig
import tabix
import torch
from torch_fftconv import FFTConv1d
from torch import nn
from torch.utils import data
import selene_sdk
import torch.nn.functional as F

NUM_REPLICATES = 12
NUM_TRAINING_SAMPLES = 40000


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train SimpleNet model on genomic data')
    
    # Required arguments
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training dataset H5 file')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset H5 file')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer (default: 0.01)')
    parser.add_argument('--validation_split', type=float, default=0.1,
                       help='Fraction of training data to use for validation (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='simplenet_trained',
                       help='Name for saving the model (default: simplenet_trained)')
    parser.add_argument('--model_num', type=str, default='001',
                       help='Model number for logging (default: 001)')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading (default: 4)')
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save models (default: ./models)')
    
    # Scheduler parameters
    parser.add_argument('--T_0', type=int, default=2,
                       help='T_0 for CosineAnnealingWarmRestarts (default: 2)')
    parser.add_argument('--T_mult', type=int, default=2,
                       help='T_mult for CosineAnnealingWarmRestarts (default: 2)')
    
    return parser.parse_args()

class H5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        self.fp = file_path
        self.data = h5py.File(file_path, 'r', libver='latest')
        self.ctr = 0
        # Get dataset length from the number of X entries
        self.length = len([k for k in self.data.keys() if k.startswith('X')])

    def __getitem__(self, idx):
        self.ctr += 1
        if self.ctr % 100000 == 0: # workaround to avoid memory issues
            self.data.close()
            self.data = h5py.File(self.fp, 'r', libver='latest')
        X = torch.Tensor(self.data['X' + str(idx)][:].T)
        Y = torch.Tensor(self.data['Y' + str(idx)][:].T)
        # For compatibility with your training loop, we return a dummy coord
        coord = 0          # (chrom, start, end, strand)
        
        #coord =[z[0].decode('utf-8') for z in coord]
        return (X, Y, coord)

    def __len__(self):
        return self.length

class SimpleNetModified(nn.Module):
    def __init__(self, input_channels=4, output_channels=4):
        super(SimpleNetModified, self).__init__()

        self.conv = nn.Conv1d(input_channels, 40, kernel_size=51, padding=25)
        self.activation = nn.Sigmoid()

        # Separate deconv layers for labels (3 channels) and SSE (1 channel)
        self.deconv_labels = FFTConv1d(40, output_channels-1, kernel_size=601, padding=300)  # 3 channels
        self.deconv_SSE = FFTConv1d(40, 1, kernel_size=601, padding=300)  # 1 channel

    def forward(self, x):
        y = self.conv(x)  # Shape: (batch_size, 40, 5000)
        yact = self.activation(y) * y  # Shape: (batch_size, 40, 5000)
        
        # Separate predictions for labels and SSE
        y_pred_label = F.softmax(self.deconv_labels(yact), dim=1)  # Shape: (batch_size, 3, 5000)
        y_pred_SSE = torch.sigmoid(self.deconv_SSE(yact))  # Shape: (batch_size, 1, 5000)
        
        # Concatenate the outputs along channel dimension
        y_pred = torch.cat([y_pred_label, y_pred_SSE], dim=1)  # Shape: (batch_size, 4, 5000)
        
        # Crop the output to match label shape: 5000 -> 4000 by removing 500 from each side
        return y_pred[:, :, 500:-500]  # Final shape: (batch_size, 4, 4000)

def multi_crossent(y_pred, y_true):
    # Multi-class cross entropy for 3 channels
    # y_pred: (batch_size, 3, 4000), y_true: (batch_size, 3, 4000)
    loss = 0
    for i in range(3):
        loss += - torch.mean(y_true[:, i, :] * torch.log(y_pred[:, i, :] + 1e-10))
    return loss / 3  # Average over channels

def bce(y_pred, y_true):
    # Binary cross entropy for the continuous label
    return nn.BCELoss()(y_pred, y_true)

def loss_function(y_pred, y_true):
    # Split the 4-channel output into 3 categorical + 1 continuous
    y_pred_split = torch.split(y_pred, [3, 1], dim=1)
    y_true_split = torch.split(y_true, [3, 1], dim=1)
    
    # Categorical loss for first 3 channels
    loss_cat = multi_crossent(y_pred_split[0], y_true_split[0]) 
    
    # Continuous loss for last channel (only where targets are >= 0)
    loss_cont = bce(y_pred_split[1][y_true_split[1] >= 0], 
                    y_true_split[1][y_true_split[1] >= 0]) 

    return loss_cat + loss_cont

# Training functions
def progress_bar(batch_idx, total_batches, message):
    bar_length = 20
    progress = float(batch_idx) / total_batches
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1:.1%} {2}".format(
        "=" * block + "-" * (bar_length - block), progress, message)
    sys.stdout.write(text)
    sys.stdout.flush()

def train(epoch, model, train_dl, optimizer, scheduler, iters, criterion):
    print('\nEpoch: %d' % epoch, flush=True)
    model.train()
    train_loss = 0
    
    for batch_idx, (inputs, targets, coord) in enumerate(train_dl):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss += float(loss)
        
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + batch_idx / iters)

        progress_bar(batch_idx, len(train_dl),
                     'Loss: %.5f' % (train_loss/(batch_idx+1)))

    return train_loss/(batch_idx+1)

def test(model, val_dl, criterion):
    model.eval()
    test_loss = 0

    for batch_idx, (inputs, targets, coord) in enumerate(val_dl):
        if batch_idx % 10 == 0:
            print("Batch", batch_idx)
            print("Input[0]:", inputs[0][500:-500, :])
            print("Target[0]:", targets[0])
            print("Coord[0]:", coord[0])
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        test_loss += float(loss)
        
        progress_bar(batch_idx, len(val_dl),
                     'Loss: %.5f' % (test_loss/(batch_idx+1)))

    return test_loss/(batch_idx+1)

def main():
    args = parse_arguments()
    
    # Print configuration
    print("Training configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print()
    
    # Load datasets
    print(f"Loading training data from {args.train_data}")
    train_dataset = H5Dataset(args.train_data)
    print(f"Loading test data from {args.test_data}")
    test_dataset = H5Dataset(args.test_data)

    # Initialize model
    model = SimpleNetModified(input_channels=4, output_channels=4)
    model.cuda()

    # Setup training components
    criterion = loss_function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    T_0 = args.T_0
    T_mult = args.T_mult
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Open log file
    log_path = os.path.join(args.save_dir, f"log.{args.model_num}.txt")
    flog_final = open(log_path, 'w')

    # Split training data for validation
    val_size = round(args.validation_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    
    print(f"Model will be saved to: {args.save_dir}")
    print(f"Log file: {log_path}")
    
    # Assuming you have train_seqs and train_tars as numpy arrays (N, T, F)
    # and you want to train 12 replicates
    best_losses = [float('inf')] * NUM_REPLICATES

    for replicate in range(1, NUM_REPLICATES):
        print(f"\n=== Starting replicate {replicate+1}/{NUM_REPLICATES} ===")
        train_ds, val_ds = data.random_split(train_dataset, (train_size, val_size))
        subset_indices = np.random.choice(len(train_ds), NUM_TRAINING_SAMPLES, replace=False)
        sampler = data.SubsetRandomSampler(subset_indices)


        print(f"Starting training with {(NUM_TRAINING_SAMPLES)} training samples and {len(val_ds)} validation samples")
        # Create data loaders
        train_dl = data.DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                num_workers=args.num_workers, pin_memory=True)
        val_dl = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, pin_memory=True)
        iters = len(train_dl)
        for epoch in range(args.epochs):
            train_loss = train(epoch, model, train_dl, optimizer, scheduler, iters, criterion)
            test_loss = test(model, val_dl, criterion)
            
            # Log results
            print(f"Epoch {epoch}: Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}")
            print(epoch, train_loss, test_loss, file=flog_final, flush=True)
            flog_final.flush()
            
            # Save best model
            if test_loss < best_losses[replicate]:
                best_losses[replicate] = test_loss
                best_model_path = os.path.join(args.save_dir, f"model.{NUM_TRAINING_SAMPLES}.rep{replicate}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with test loss: {best_losses[replicate]:.5f}")

    flog_final.close()
    print("Training finished")

if __name__ == "__main__":
    main()