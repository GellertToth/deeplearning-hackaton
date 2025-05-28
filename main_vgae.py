import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from src.loadData import GraphDataset, PreloadedGraphDataset, load_data
from src.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import torch.nn as nn
from src.models import VGAE
from torch.optim.lr_scheduler import CosineAnnealingLR
import hashlib
from sklearn.metrics import f1_score
import final_models

def string_to_int(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (1_000_000_007)

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

import torch
import torch.nn.functional as F
# {0: 0.12101063829787234,  1: 0.17039007092198583, 2: 0.2931737588652482, 3: 0.17553191489361702, 4: 0.17411347517730497, 5: 0.06578014184397163}
class_probs = torch.tensor([0.15, 0.16, 0.25, 0.16, 0.16, 0.12])
inv_freq = 1.0 / class_probs
weights = inv_freq / inv_freq.sum()

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
    predictions, true_labels = [], []
    p_dist=torch.zeros((6, ))
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            _, _, _, output = model(data, inference=True)
            pred = output.argmax(dim=1)
            p_dist += F.one_hot(pred, num_classes=6).float().cpu().mean(dim=0)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(data.y.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    f1 = f1_score(true_labels, predictions, average='weighted')  
    
    print(f"Pred distribution: {(p_dist / len(data_loader)).tolist()}")
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, f1, predictions
    return predictions

def evaluate_models(data_loader, models, weights, device, calculate_accuracy=False):
    for model in models:
        model.eval()
    correct = 0
    total = 0
    predictions = []
    p_dist=torch.zeros((6, ))
    weights = weights[:, None, None].to(device)
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            preds = (torch.stack([model(data, inference=True)[3] for model in models], dim=0) * weights).sum(dim=0)
            pred = preds.argmax(dim=1)
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


def plot_training_progress(train_losses, train_accuracies, output_dir, model_id):
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
    plt.savefig(os.path.join(output_dir, f"training_progress_{model_id}.png"))
    plt.close()

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.T = 1

    def forward(self, input):
        z, mu, logvar, logits = self.model(input)
        return z, mu, logvar, logits / self.T

    def set_temperature(self, data_loader, device):
        print(f"Setting temperature")
        self.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
                    data = data.to(device)
                    _, _, _, logits = self.model(data)
                    predictions.append(logits.cpu())
                    true_labels.append(data.y.cpu())

        logits = torch.cat(predictions)
        labels = torch.cat(true_labels)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.T = self.temperature.item()
        print(f"Optimal temperature: {self.temperature.item()}")


def main(args):
    seed = string_to_int(args.model_id)
    set_seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hidden_dim = 128
    emb_dim = 16
    warmup_epochs=10
    initial_lr = 1e-5
    target_lr = 2.5 * 1e-4
    minimum_lr = 1e-6
    
    num_epochs = 200
    batch_size = args.batch_size
    num_checkpoints = 5

    device = f"cuda:{args.device}"
    print(f"Using device {device}")

    # Identify dataset folder (A, B, C, or D)
    if args.pretraining:
        test_dir_name = "pretraining"
    else:
        test_dir_name = os.path.basename(os.path.dirname(args.train_path if (not args.train_path is None) else args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, f"training_{args.model_id}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well


    # Define checkpoint path relative to the script's directory
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)


    if args.train_path:
        model = VGAE(in_channels=1, edge_attr_dim=7, hidden_dim=hidden_dim, latent_dim=emb_dim, num_classes=6).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=target_lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs, eta_min=minimum_lr)
        # If train_path is provided, train the model
        if not args.pretrained_path is None:
            model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        if args.pretraining:
            train_graphs, val_graphs = load_data(args.train_path, round=0, n_folds=args.n_folds, train_folds_to_use=args.train_folds_to_use, test_size=0.2, seed=seed)
        else:
            train_graphs, val_graphs = load_data(args.train_path, round=0, n_folds=1, train_folds_to_use=1, test_size=0.2, seed=seed)
        train_dataset = PreloadedGraphDataset(train_graphs, transform=add_zeros)
        val_dataset = PreloadedGraphDataset(val_graphs, transform=add_zeros)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


        # Training loop
        best_f1 = 0.0
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

            if args.pretraining and epoch % 10 == 0 and epoch != 0:
                train_graphs, val_graphs = load_data(args.train_path, round=(epoch//10), n_folds=args.n_folds, train_folds_to_use=args.train_folds_to_use, test_size=0.2, seed=seed)
                train_dataset = PreloadedGraphDataset(train_graphs, transform=add_zeros)
                val_dataset = PreloadedGraphDataset(val_graphs, transform=add_zeros)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            train_loss = train(
                train_loader, model, optimizer, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}_round_{args.model_id}"),
                current_epoch=epoch
            )
            train_acc, f1, _ = evaluate(val_loader, model, device, calculate_accuracy=True)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Val Acc: {train_acc:.4f}, Val f1: {f1:.4f}")
            # Save logs for training progress
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Val Acc: {train_acc:.4f}, Val f1: {f1:.4f}")

            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_round_{args.model_id}_f1_{f1:.4f}_best.pth")
                torch.save(model.state_dict(), path)
                print(f"Best model updated and saved at {path}")

        # Plot training progress in current directory
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"), args.model_id)

    if not args.test_path is None:
        print("Predicting")
        model_paths = final_models.final_models[test_dir_name]
        models = []
        weights = []
        for path in model_paths:
            print(f"Loading model from {path}")
            model = VGAE(in_channels=1, edge_attr_dim=7, hidden_dim=hidden_dim, latent_dim=emb_dim, num_classes=6).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            def get_f1_score(model_name):
                return float(model_name.split("_")[-2])
            f1 = get_f1_score(path)
            weights.append(f1)
            models.append(model)

        weights = torch.tensor(weights)
        weights = weights / weights.sum()
        print(f"Model weights according to val f1 {weights}")
        test_dataset = GraphDataset(args.test_path, transform=add_zeros)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        predictions = evaluate_models(test_loader, models, weights, device, calculate_accuracy=False)
        save_predictions(predictions, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, help="Path to the test dataset.")
    parser.add_argument("--pretrained_path", type=str, help="Path to the pretrained model.")
    parser.add_argument("--pretraining", type=bool, help="Pretrain or finetune", default=False)
    parser.add_argument("--model_id", type=str, help="Model id to enable training more than one mode", default="model0")
    parser.add_argument("--n_folds", type=int, help="Number of folds", default=3)
    parser.add_argument("--train_folds_to_use", type=int, help="Train folds to use together", default=1)
    parser.add_argument("--device", type=int, default=0, help="GPU device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    main(args)


