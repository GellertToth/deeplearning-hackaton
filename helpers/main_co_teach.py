import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from source.loadData import GraphDataset
from source.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from source.models import GNN 

# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

import torch
import torch.nn.functional as F


def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
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
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCELoss(nn.Module):
    def __init__(self, q=0.7, num_classes=None):
        """
        Generalized Cross Entropy Loss
        Args:
            q: exponent hyperparameter, controls sensitivity to noise. 0.7 is a good default.
            num_classes: number of classes in your classification problem.
            reduction: 'mean' or 'sum'
        """
        super(GCELoss, self).__init__()
        assert q > 0 and q <= 1, "q should be in (0, 1]"
        self.q = q
        self.num_classes = num_classes

    def forward(self, logits, targets, reduction="mean"):
        """
        logits: [batch_size, num_classes] (raw output from the model)
        targets: [batch_size] (ground-truth labels)
        """
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Get p_y^q for each sample
        pt = (probs * targets_one_hot).sum(dim=1)
        loss = (1 - pt ** self.q) / self.q

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
LAMBDA = 1.5

def cotraining_batch_step(model1, model2, opt1, opt2, criterion, scaler1, scaler2, batch, warm_up):
    opt1.zero_grad()
    opt2.zero_grad()
    with autocast():
        out1 = model1(batch)
        out2 = model2(batch)
        loss1_raw = criterion(out1, batch.y, reduction='none')
        loss2_raw = criterion(out2, batch.y, reduction='none')
    if warm_up:   
        scaler1.scale(loss1_raw.mean()).backward(); scaler1.step(opt1), scaler1.update()
        scaler2.scale(loss2_raw.mean()).backward(); scaler2.step(opt2), scaler2.update()
        return loss1_raw.mean().item(), loss2_raw.mean().item()
    
    with torch.no_grad():
        probs1 = torch.softmax(out1, dim=1)
        probs2 = torch.softmax(out2, dim=1)
        targets_one_hot = F.one_hot(batch.y, num_classes=6).float()
        pt_model1 = (probs1 * targets_one_hot).sum(dim=1)
        pt_model2 = (probs2 * targets_one_hot).sum(dim=1)

    scaled_loss1 = (torch.exp(- LAMBDA * pt_model2.detach()) * loss1_raw).mean()
    scaled_loss2 = (torch.exp(- LAMBDA * pt_model1.detach()) * loss2_raw).mean()

    scaler1.scale(scaled_loss1).backward(); scaler1.step(opt1), scaler1.update()
    scaler2.scale(scaled_loss2).backward(); scaler2.step(opt2), scaler2.update()

    return scaled_loss1.item(), scaled_loss2.item()

def cotrain(data_loader, model1, model2, optimizer1, optimizer2, criterion, scaler1, scaler2, device, save_checkpoints, checkpoint_path, current_epoch, warm_up):
    model1.train()
    model2.train()

    total_loss = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        loss1, loss2 = cotraining_batch_step(model1, model2, optimizer1, optimizer2, criterion, scaler1, scaler2, data.to(device), warm_up)
        total_loss += loss1

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model1.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader)


def main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 5
    

    # if args.gnn == 'gin':
    #     model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    # elif args.gnn == 'gin-virtual':
    #     model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    # elif args.gnn == 'gcn':
    #     model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    # elif args.gnn == 'gcn-virtual':
    #     model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    # else:
    #     raise ValueError('Invalid GNN type')
    model = GNN(gnn_type="gin", num_class=6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, graph_pooling="attention").to(device)
    model1 = GNN(gnn_type="gin", num_class=6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, graph_pooling="attention").to(device)
    model2 = GNN(gnn_type="gin", num_class=6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, graph_pooling="attention").to(device)
    
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.0001)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0001)
    scaler1 = GradScaler()
    scaler2 = GradScaler()
    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well


    # Define checkpoint path relative to the script's directory
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # If train_path is provided, train the model
    if args.train_path:
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Training loop
        num_epochs = args.epochs
        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []

        # Calculate intervals for saving checkpoints
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        forget_rate = 0.2
        remember_schedule = [1 - forget_rate * min(1, epoch / num_epochs) for epoch in range(num_epochs)]
        # q_start, q_end = 1, 0.5
        for epoch in range(num_epochs):
            # q = q_start - (q_start - q_end) * (epoch / (num_epochs-1))
            warm_up = epoch < 3 
            criterion = GCELoss(num_classes=6, q=0.4)
            # train_loss = train(
            #     train_loader, model, optimizer, criterion, device,
            #     save_checkpoints=(epoch + 1 in checkpoint_intervals),
            #     checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
            #     current_epoch=epoch
            # )
            # remember_rate = 1.0 if epoch < 2 else remember_schedule[epoch]

            train_loss = cotrain(train_loader, model1, model2, optimizer1, optimizer2, criterion, scaler1, scaler2, device, 
                                 save_checkpoints=(epoch + 1 in checkpoint_intervals), 
                                 checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"), 
                                 current_epoch=epoch, 
                                 warm_up=warm_up)
            
            train_acc, _ = evaluate(train_loader, model1, device, calculate_accuracy=True)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Save logs for training progress
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Save best model
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                torch.save(model1.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")

        # Plot training progress in current directory
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

    # Generate predictions for the test set using the best model
    model.load_state_dict(torch.load(checkpoint_path))
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=3, help='number of GNN message passing layers (default: 3)')
    parser.add_argument('--emb_dim', type=int, default=256, help='dimensionality of hidden units in GNNs (default: 128)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    
    args = parser.parse_args()
    main(args)
