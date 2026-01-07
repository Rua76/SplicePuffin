import os
import sys

import argparse
import numpy as np

import torch
from torch import nn
from torch.utils import data
from matplotlib import pyplot as plt

from models import H5Dataset, SimpleNetModified_DA_SSE, SimpleNetModified_DA_TripleLayers
from utils import evaluate_model_auprc, evaluate_model_regression, plot_regression_results, plot_auprc_results, progress_bar

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train SimpleNet model on genomic data')
    
    # Required arguments
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training dataset H5 file')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset H5 file')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
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

# ------------------------------------------------------------
# loss functions with masking
# ------------------------------------------------------------

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

def masked_kl_loss(y_pred, y_true, mask):
    """
    Masked KL divergence loss.
    y_pred: [batch, 2, seq_len]
    y_true: [batch, 2, seq_len]
    mask:   [batch, seq_len]
    """
    eps = 1e-10
    # 1. Normalize along the sequence dimension
    #    (so each channel is a probability distribution)
    pred = y_pred + eps
    pred = pred / pred.sum(dim=2, keepdim=True)
    target = y_true + eps
    target = target / target.sum(dim=2, keepdim=True)

    # 2. Compute KL per position (batch, 2, seq_len)
    kl = target * (torch.log(target + eps) - torch.log(pred + eps))
    mask_expanded = mask.unsqueeze(1).expand_as(kl)
    masked_kl = (kl * mask_expanded).sum() / (mask_expanded.sum() + eps)

    return masked_kl

def loss_function(y_pred, y_true, mask):
    """
    Modified loss function that uses masking
    y_pred: [batch, 2, seq_len] - model predictions (donor, acceptor)
    y_true: [batch, 2, seq_len] - ground truth labels
    mask: [batch, 1, seq_len] - mask from dataset
    """
    return masked_kl_loss(y_pred, y_true, mask)

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
        

        progress_bar(batch_idx, len(train_dl),
                     'Loss: %.5f' % (train_loss/(batch_idx+1)))
    if scheduler is not None:
        scheduler.step()


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
    replicate_dir = args.save_dir + f"/replicate_{replicate}/"
    os.makedirs(replicate_dir, exist_ok=True)

    log_path = os.path.join(replicate_dir, f"log.rep{replicate}.txt")
    flog_final = open(log_path, 'w')

    #print("Replicate\tDonor_auPRC\tAcceptor_auPRC\tCombined_auPRC", file=flog_auprc)
    flog_regression = open(os.path.join(replicate_dir, f"regression_metrics_rep{replicate}.log"), "w")
    # --- Write header line for regression metrics log ---
    print(
        "Replicate\tEpoch\t"
        "Donor_MSE\tDonor_R2\tDonor_Pearson\t"
        "Acceptor_MSE\tAcceptor_R2\tAcceptor_Pearson\t"
        "Combined_MSE\tCombined_R2\tCombined_Pearson",
        file=flog_regression,
        flush=True,
    )
    # --- train/val split ---
    val_size = round(args.validation_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = data.random_split(train_dataset, (train_size, val_size))
    train_dl = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_dl = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- model setup ---
    model = SimpleNetModified_DA_SSE(input_channels=4).cuda()
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

        # --- Save best model ---
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (rep {replicate}, epoch {epoch})", flush=True)

        # --- Every 100 epochs: evaluate regression metrics using best model ---
        if (epoch + 1) % 100 == 0:
            print(f"\n=== Epoch {epoch + 1}: Evaluating regression metrics for best model so far ===", flush=True)
            
            # Load best model
            best_model_state = torch.load(best_model_path, map_location="cpu")
            model.load_state_dict(best_model_state)

            # Compute regression metrics
            reg_results = evaluate_model_regression(model, test_dl, replicate)
            donor = reg_results["Donor"]
            acceptor = reg_results["Acceptor"]
            combined = reg_results["Combined"]

            # Log results
            print(
                f"[Epoch {epoch + 1}] "
                f"Donor(MSE={donor['MSE']:.6f}, R²={donor['R2']:.6f}, P={donor['Pearson']:.4f}) | "
                f"Acceptor(MSE={acceptor['MSE']:.6f}, R²={acceptor['R2']:.6f}, P={acceptor['Pearson']:.4f}) | "
                f"Combined(MSE={combined['MSE']:.6f}, R²={combined['R2']:.6f}, P={combined['Pearson']:.4f})",
                flush=True,
            )

            print(
                f"{replicate}\t{epoch + 1}\t"
                f"{donor['MSE']:.6f}\t{donor['R2']:.6f}\t{donor['Pearson']:.6f}\t"
                f"{acceptor['MSE']:.6f}\t{acceptor['R2']:.6f}\t{acceptor['Pearson']:.6f}\t"
                f"{combined['MSE']:.6f}\t{combined['R2']:.6f}\t{combined['Pearson']:.6f}",
                file=flog_regression,
                flush=True,
            )

    # --- final evaluation ---
    model.load_state_dict(torch.load(best_model_path))
    final_results = evaluate_model_regression(model, test_dl, replicate)
    donor, acceptor, combined = (
        final_results["Donor"],
        final_results["Acceptor"],
        final_results["Combined"],
    )

    print(f"\nReplicate {replicate} Final Results:")
    print(f"  Donor:    MSE={donor['MSE']:.6f}, R²={donor['R2']:.6f}, Pearson={donor['Pearson']:.6f}")
    print(f"  Acceptor: MSE={acceptor['MSE']:.6f}, R²={acceptor['R2']:.6f}, Pearson={acceptor['Pearson']:.6f}")
    print(f"  Combined: MSE={combined['MSE']:.6f}, R²={combined['R2']:.6f}, Pearson={combined['Pearson']:.6f}")

    print(
        f"{replicate}\tFINAL\t"
        f"{donor['MSE']:.6f}\t{donor['R2']:.6f}\t{donor['Pearson']:.6f}\t"
        f"{acceptor['MSE']:.6f}\t{acceptor['R2']:.6f}\t{acceptor['Pearson']:.6f}\t"
        f"{combined['MSE']:.6f}\t{combined['R2']:.6f}\t{combined['Pearson']:.6f}",
        file=flog_regression,
        flush=True,
    )

    flog_final.close()
    flog_regression.close()
    print(f"Finished replicate {replicate}")



if __name__ == "__main__":
    main()
