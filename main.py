import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
# Load utility functions from cloned repository

import sys
import os

# Add the parent directory of src to the system path
sys.path.append(os.path.abspath('./'))

from src.loadData import GraphDataset
from src.utils import set_seed
from src.models import GNNEncoderDecoder, EnsembleModel
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set the random seed
set_seed()


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        node_emb, output = model(data)
        loss = model.loss(node_emb, output, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader),  correct / total

from sklearn.metrics import f1_score

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0
    true_labels = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            node_emb, output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            
            if calculate_accuracy:
                true_labels.extend(data.y.cpu().numpy())
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    if calculate_accuracy:
        f1 = f1_score(true_labels, predictions, average='weighted')  
        accuracy = correct / total
        return  total_loss / len(data_loader), accuracy, f1
    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.getcwd() 
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

    # Plot losscolabes, label="Training Loss", color='blue')
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

def train_once(model, args, voter, full_dataset, test_dir_name, logs_folder, script_dir, device):
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(voter+1)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_voter_{voter}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3    

    initial_lr = 1e-5
    target_lr = 1e-3
    minimum_lr = 1e-6

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs-args.warmup_epochs, eta_min=minimum_lr)

    num_epochs = args.epochs
    best_f1 = 0.0   

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    if num_checkpoints > 1:
        checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
    else:
        checkpoint_intervals = [num_epochs]

    def get_lr(epoch):
        if epoch < args.warmup_epochs:
            lr = initial_lr + (target_lr - initial_lr) * (epoch / args.warmup_epochs)
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[-1]
        return lr


    for epoch in range(num_epochs):
        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Current learning rate: {lr}")

        train_loss, train_acc = train(
            train_loader, model, optimizer, device,
            save_checkpoints=(epoch + 1 in checkpoint_intervals),
            checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}_voter_{voter}"),
            current_epoch=epoch
        )

        val_loss,val_acc, f1 = evaluate(val_loader, model, device, calculate_accuracy=True)

        print(f"Voter {voter}, Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val f1: {f1:.4f}")
        logging.info(f"Voter {voter}, Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val f1: {f1:.4f}")
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model updated and saved at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, f"plots_voter_{voter}"))
    plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, f"plotsVal_voter_{voter}"))

    return model, best_f1


def main(args):
    script_dir = os.getcwd() 
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    def get_model():
        if args.gnn == 'gin':
            model = GNNEncoderDecoder(gnn_type='gin', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False, noise_prob=args.noise_prob).to(device)
        elif args.gnn == 'gin-virtual':
            model = GNNEncoderDecoder(gnn_type='gin', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True, noise_prob=args.noise_prob).to(device)
        elif args.gnn == 'gcn':
            model = GNNEncoderDecoder(gnn_type='gcn', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False, noise_prob=args.noise_prob).to(device)
        elif args.gnn == 'gcn-virtual':
            model = GNNEncoderDecoder(gnn_type='gcn', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True, noise_prob=args.noise_prob).to(device)
        else:
            raise ValueError('Invalid GNN type')
        return model

    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, f"training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)
    if args.train_path:
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)

        models, weights = [], []
        for voter in range(args.num_voters):
            model, weight = train_once(get_model(), args, voter, full_dataset, test_dir_name, logs_folder, script_dir, device)
            models.append(model)
            weights.append(weight)
        ensemble = EnsembleModel(models, weights, device)

        torch.save(ensemble.state_dict(), checkpoint_path)

    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = EnsembleModel([get_model() for _ in range(args.num_voters)], [1 for _ in range(args.num_voters)], device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)
    

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, help="Path to the test dataset.")

    parser.add_argument("--device", type=int, default=0, help="GPU device to use")
    parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-gnn", type=str, default="gin")
    parser.add_argument("--drop_ratio", type=float, default=0.3, help="Drop ratio")

    parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_layer", type=int, default=3, help="Number of layers")

    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Number of warmup epochs")

    parser.add_argument("--noise_prob", type=float, default=0.2, help="Noise prob")

    parser.add_argument("--num_voters", type=int, default=2, help="Number of voters to train")

    args = parser.parse_args()
    main(args)



def get_arguments():    

    
    # args['noise_prob'] = get_user_input("Noise probability p (used if baseline_mode=2)", default=0.2, type_cast=float)
    args['noise_prob'] = 0.2

    args["num_voters"] = 2


    
    return argparse.Namespace(**args)
