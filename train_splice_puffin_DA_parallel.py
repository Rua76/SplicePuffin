import os
import sys

import argparse
import numpy as np

import torch
from torch import nn
from torch.utils import data
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt

from models import H5Dataset, SimpleNetModified_DA

NUM_REPLICATES = 6

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train SimpleNet model on genomic data')
    
    # Required arguments
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training dataset H5 file')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset H5 file')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer (default: 0.01)')
    parser.add_argument('--validation_split', type=float, default=0.1,
                       help='Fraction of training data to use for validation (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=100,
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
    parser.add_argument('--test_mode', action='store_true',
                       help='If set, run in test mode with limited data')
    parser.add_argument('--replicate_id', type=int, default=0,
                    help='Replicate ID for parallel training (default: 0)')

    return parser.parse_args()

# -------------------------------
# Plotting function
# -------------------------------
def plot_auprc_results(all_results, save_dir):
    """Plot auPRC results for all replicates using line plots"""
    if not all_results:
        print("No results to plot.")
        return
    
    # Extract data for plotting
    replicates = list(range(len(all_results)))
    donor_scores = [result["task1_per_class"][0] for result in all_results]
    acceptor_scores = [result["task1_per_class"][1] for result in all_results]
    mean_scores = [result["task1_mean"] for result in all_results]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot lines with markers
    plt.plot(replicates, donor_scores, marker='o', linewidth=2.5, markersize=8, 
             label='Donor auPRC', color='royalblue', alpha=0.8)
    plt.plot(replicates, acceptor_scores, marker='s', linewidth=2.5, markersize=8, 
             label='Acceptor auPRC', color='crimson', alpha=0.8)
    plt.plot(replicates, mean_scores, marker='^', linewidth=2.5, markersize=8, 
             label='Mean auPRC', color='forestgreen', alpha=0.8)
    
    plt.xlabel('Replicate Number')
    plt.ylabel('auPRC Score')
    plt.title('auPRC Performance Across Replicates')
    plt.xticks(replicates, [f'Rep {i}' for i in replicates])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, "auprc_replicates_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {plot_path}")

# Training functions
def progress_bar(batch_idx, total_batches, message):
    bar_length = 20
    progress = float(batch_idx) / total_batches
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1:.1%} {2}".format(
        "=" * block + "-" * (bar_length - block), progress, message)
    sys.stdout.write(text)
    sys.stdout.flush()


def masked_bce_loss(y_pred, y_true, mask):
    """
    Binary cross entropy loss with masking
    y_pred: [batch, 2, seq_len] - model predictions
    y_true: [batch, 2, seq_len] - ground truth labels  
    mask: [batch, seq_len] - 1 for valid, 0 for masked positions
    """
    # Calculate BCE loss for all positions
    bce_loss = nn.BCELoss(reduction='none')(y_pred, y_true)  # [batch, 2, seq_len]
    
    # Expand mask from [batch, seq_len] to [batch, 2, seq_len]
    mask_expanded = mask.unsqueeze(1).expand_as(bce_loss)  # Add channel dimension and expand
    
    # Apply mask and normalize by number of valid positions
    masked_loss = (bce_loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
    
    return masked_loss

def loss_function(y_pred, y_true, mask):
    """
    Modified loss function that uses masking
    y_pred: [batch, 2, seq_len] - model predictions (donor, acceptor)
    y_true: [batch, 2, seq_len] - ground truth labels
    mask: [batch, 1, seq_len] - mask from dataset
    """
    return masked_bce_loss(y_pred, y_true, mask)

def train(epoch, model, train_dl, optimizer, scheduler, iters, criterion):
    print('\nEpoch: %d' % epoch, flush=True)
    model.train()
    train_loss = 0
    
    for batch_idx, (inputs, targets, mask, coord) in enumerate(train_dl):
        inputs, targets, mask = inputs.cuda(), targets.cuda(), mask.cuda()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets, mask)
        train_loss += float(loss)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar(batch_idx, len(train_dl),
                     'Loss: %.5f' % (train_loss/(batch_idx+1)))

    return train_loss/(batch_idx+1)

def test(model, val_dl, criterion):
    model.eval()
    test_loss = 0

    for batch_idx, (inputs, targets, mask, coord) in enumerate(val_dl):
        inputs, targets, mask = inputs.cuda(), targets.cuda(), mask.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets, mask)
        test_loss += float(loss)
        
        progress_bar(batch_idx, len(val_dl),
                     'Loss: %.5f' % (test_loss/(batch_idx+1)))

    return test_loss/(batch_idx+1)

def evaluate_model_auprc(model, test_dl, replicate_num):
    """
    Evaluate model and compute auPRC for Donor and Acceptor channels with masking
    """
    model.eval()
    all_outputs = []
    all_targets = []
    all_masks = []
    
    print(f"\n=== Evaluating Replicate {replicate_num} on Test Set ===")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, M, coord) in enumerate(test_dl):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(test_dl)}")
                
            inputs, targets, M = inputs.cuda(), targets.cuda(), M.cuda()
            outputs = model(inputs)
            
            # Store predictions, targets, and masks
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(M.cpu().numpy())
    
    # Concatenate all batches
    all_outputs = np.concatenate(all_outputs, axis=0)  # shape: (N, 2, 4000)
    all_targets = np.concatenate(all_targets, axis=0)  # shape: (N, 2, 4000)
    all_masks = np.concatenate(all_masks, axis=0)      # shape: (N, 4000)
    
    print(f"Collected predictions: {all_outputs.shape}")
    print(f"Collected targets: {all_targets.shape}")
    print(f"Collected masks: {all_masks.shape}")
    
    # Calculate auPRC for each channel
    auprc_results = {}
    
    for channel_idx, channel_name in enumerate(['Donor', 'Acceptor']):
        print(f"\nComputing auPRC for {channel_name} channel...")
        
        # Get predictions and targets for this channel
        y_scores = all_outputs[:, channel_idx, :]  # shape: (N, 4000)
        y_true = all_targets[:, channel_idx, :]    # shape: (N, 4000)
        
        # Apply mask: only keep positions where mask == 1
        valid_positions = all_masks == 1
        
        # Flatten only the valid positions
        y_scores_valid = y_scores[valid_positions]
        y_true_valid = y_true[valid_positions]
        
        print(f"  Total positions: {np.prod(y_true.shape)}")
        print(f"  Valid positions: {len(y_true_valid)}")
        print(f"  Masked positions: {np.prod(y_true.shape) - len(y_true_valid)}")
        print(f"  Positive samples: {np.sum(y_true_valid == 1)}")
        print(f"  Negative samples: {np.sum(y_true_valid == 0)}")
        print(f"  Positive rate: {np.sum(y_true_valid == 1) / len(y_true_valid):.6f}")
        
        # Only compute auPRC if we have both positive and negative samples
        if len(np.unique(y_true_valid)) < 2:
            print(f"  WARNING: Only one class present in {channel_name} channel")
            auprc_results[channel_name] = 0.0
            continue
            
        # Calculate precision-recall curve and auPRC
        precision, recall, thresholds = precision_recall_curve(y_true_valid, y_scores_valid)
        auprc = auc(recall, precision)
        
        auprc_results[channel_name] = auprc
        print(f"  {channel_name} auPRC: {auprc:.6f}")
    
    # Calculate combined auPRC (micro-average across both channels)
    print(f"\nComputing combined auPRC...")
    
    # Method 1: Process each channel separately and combine
    combined_scores = []
    combined_labels = []
    
    for channel_idx in range(2):
        y_scores = all_outputs[:, channel_idx, :]  # shape: (N, 4000)
        y_true = all_targets[:, channel_idx, :]    # shape: (N, 4000)
        valid_positions = all_masks == 1
        
        # Flatten only the valid positions for this channel
        y_scores_valid = y_scores[valid_positions]
        y_true_valid = y_true[valid_positions]
        
        combined_scores.append(y_scores_valid)
        combined_labels.append(y_true_valid)
    
    # Concatenate across both channels
    y_scores_combined_valid = np.concatenate(combined_scores)
    y_true_combined_valid = np.concatenate(combined_labels)
    
    print(f"  Combined valid positions: {len(y_true_combined_valid)}")
    print(f"  Combined positive samples: {np.sum(y_true_combined_valid == 1)}")
    print(f"  Combined negative samples: {np.sum(y_true_combined_valid == 0)}")
    print(f"  Combined positive rate: {np.sum(y_true_combined_valid == 1) / len(y_true_combined_valid):.6f}")
    
    if len(np.unique(y_true_combined_valid)) >= 2:
        precision_combined, recall_combined, _ = precision_recall_curve(y_true_combined_valid, y_scores_combined_valid)
        auprc_combined = auc(recall_combined, precision_combined)
        auprc_results['Combined'] = auprc_combined
        print(f"Combined auPRC: {auprc_combined:.6f}")
    else:
        print(f"WARNING: Only one class present in combined data")
        auprc_results['Combined'] = 0.0
    
    return auprc_results

def main():
    args = parse_arguments()
    replicate = args.replicate_id
    torch.manual_seed(1000 + replicate)
    np.random.seed(1000 + replicate)

    # Print configuration
    print(f"\n=== Starting replicate {replicate} ===")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print()

    # Load data
    train_dataset = H5Dataset(args.train_data)
    test_dataset = H5Dataset(args.test_data)
    test_dl = data.DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- unique output directory for this replicate ---
    replicate_dir = args.save_dir
    os.makedirs(replicate_dir, exist_ok=True)

    log_path = os.path.join(replicate_dir, f"log.rep{replicate}.txt")
    flog_final = open(log_path, 'w')

    auprc_log_path = os.path.join(replicate_dir, f"auprc.rep{replicate}.txt")
    flog_auprc = open(auprc_log_path, 'w')
    print("Replicate\tDonor_auPRC\tAcceptor_auPRC\tCombined_auPRC", file=flog_auprc)

    # --- train/val split ---
    val_size = round(args.validation_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = data.random_split(train_dataset, (train_size, val_size))
    train_dl = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_dl = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- model setup ---
    model = SimpleNetModified_DA(input_channels=4).cuda()
    criterion = loss_function
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=50)
    iters = len(train_dl)
    best_loss = float('inf')
    best_model_path = os.path.join(replicate_dir, f"model.rep{replicate}.pth")

    # --- training loop ---
    for epoch in range(args.epochs):
        train_loss = train(epoch, model, train_dl, optimizer, scheduler, iters, criterion)
        val_loss = test(model, val_dl, criterion)
        print(f"Epoch {epoch}: Train {train_loss:.5f}, Val {val_loss:.5f}", flush=True)
        print(epoch, train_loss, val_loss, file=flog_final, flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (rep {replicate}, epoch {epoch})")

    # --- evaluate ---
    model.load_state_dict(torch.load(best_model_path))
    auprc_results = evaluate_model_auprc(model, test_dl, replicate)
    print(f"\nReplicate {replicate} results:")
    print(f"Donor={auprc_results['Donor']:.6f}, Acceptor={auprc_results['Acceptor']:.6f}, Combined={auprc_results['Combined']:.6f}")
    print(f"{replicate}\t{auprc_results['Donor']:.6f}\t{auprc_results['Acceptor']:.6f}\t{auprc_results['Combined']:.6f}",
          file=flog_auprc, flush=True)

    flog_final.close()
    flog_auprc.close()
    print(f"Finished replicate {replicate}")


if __name__ == "__main__":
    main()