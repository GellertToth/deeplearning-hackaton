import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from src.loadData import GraphDataset, GraphDatasetDownsample
from src.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import torch.nn as nn
from src.models_new import VGAE
from torch.optim.lr_scheduler import CosineAnnealingLR


# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

import torch
import torch.nn.functional as F

def train(data_loader, model, optimizer, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        z, mu, logvar, class_logits = model(data)
        loss = model.loss(z, mu, logvar, class_logits, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader)

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    p_dist=torch.zeros((6, ))
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            _, _, _, output = model(data)
            pred = output.argmax(dim=1)
            p_dist += F.one_hot(pred, num_classes=6).float().cpu().mean(dim=0)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    print(f"Pred distribution: {(p_dist / len(data_loader)).tolist()}")
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()

def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hidden_dim = 128
    emb_dim = 16
    warmup_epochs=10
    initial_lr = 1e-5
    target_lr = 2.5 * 1e-4
    minimum_lr = 1e-6
    
    num_epochs = 100
    batch_size = 32
    num_checkpoints = 5

    device = "cuda:0"

    model = VGAE(in_channels=1, edge_attr_dim=7, hidden_dim=hidden_dim, latent_dim=emb_dim, num_classes=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=target_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs, eta_min=minimum_lr)

    # Identify dataset folder (A, B, C, or D)
    if args.pretraining:
        test_dir_name = "pretraining"
    else:
        test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well


    # Define checkpoint path relative to the script's directory
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_{args.model_id}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # If train_path is provided, train the model
    if not args.pretrained_path is None:
        model.load_state_dict(torch.load(args.pretrained_path))

    if args.train_path:
        if args.pretraining:
            train_dataset = GraphDatasetDownsample(args.train_path, transform=add_zeros, subset_ratio=args.subset_ratio, round=0)
        else:
            train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []

        # Calculate intervals for saving checkpoints
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        def get_lr(epoch):
            if epoch < warmup_epochs:
                lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
            else:
                scheduler.step()
                lr = scheduler.get_last_lr()[-1]
            return lr
            

        for epoch in range(num_epochs):
            lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Current learning rate: {lr}")

            if args.pretraining and epoch % 10 == 0 and epoch != 0 and args.subset_ratio != 1:
                train_dataset = GraphDatasetDownsample(args.train_path, transform=add_zeros, subset_ratio=args.subset_ratio, round=epoch//10)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            train_loss = train(
                train_loader, model, optimizer, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}_{args.model_id}"),
                current_epoch=epoch
            )
            train_acc, _ = evaluate(train_loader, model, device, calculate_accuracy=True)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Save logs for training progress
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Save best model
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")

        # Plot training progress in current directory
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

    if not args.pretraining:
        test_dataset = GraphDataset(args.test_path, transform=add_zeros)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.load_state_dict(torch.load(checkpoint_path))
        predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
        save_predictions(predictions, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, help="Path to the test dataset.")
    parser.add_argument("--pretrained_path", type=str, help="Path to the pretrained model.")
    parser.add_argument("--pretraining", type=bool, help="Pretrain or finetune", default=False)
    parser.add_argument("--model_id", type=str, help="Model id to enable training more than one mode", default="model0")
    parser.add_argument("--subset_ratio", type=float, help="Percentage of data to load when pretraining", default=0.4)




    args = parser.parse_args()
    main(args)


