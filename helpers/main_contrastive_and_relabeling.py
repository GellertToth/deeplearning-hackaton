# %%
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
import torch.nn as nn
from source.models import GNN, MLP, CompleteModel

# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

import torch
import torch.nn.functional as F

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


class GCELoss(nn.Module):
    def __init__(self, q=0.7, smoothing=0.1, temperature=2.0, num_classes=6):
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
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, logits, targets, reduction="mean"):
        """
        logits: [batch_size, num_classes] (raw output from the model)
        targets: [batch_size] (ground-truth labels)
        """
        probs = F.softmax(logits / self.temperature, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        targets_one_hot = (1 - self.smoothing) * targets_one_hot + self.smoothing / self.num_classes

        # Get p_y^q for each sample
        pt = (probs * targets_one_hot).sum(dim=1)
        loss = (1 - pt ** self.q) / self.q

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = x.mean(dim=0)
        return -(x * torch.log(x + 1e-6)).sum()


def compute_embeddings_and_preds(dataloader, embedding_model, classifier, device='cuda'):
    embedding_model.eval();  classifier.eval()

    all_embeddings = []
    all_pred_probs = []
    original_labels = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = data.to(device)
            emb = embedding_model(data) 
            logits = classifier(emb)
            probs = torch.softmax(logits, dim=1)

            all_embeddings.append(emb.cpu())
            all_pred_probs.append(probs.cpu())
            original_labels.append(data.cpu().y)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_pred_probs = torch.cat(all_pred_probs, dim=0) 
    original_labels = torch.cat(original_labels, dim=0)

    return all_embeddings, all_pred_probs, original_labels

def neighbor_aware_label_correction(embeddings, pred_probs, original_labels, mu=0.8, K=5, delta_prime=0.9, c=1, cmax=10, tau=0.5):
    print(f"Avg max pred {pred_probs.max(dim=1).values.mean().item()}")
    emb_norm = F.normalize(embeddings, p=2, dim=1) 
    sim_matrix = torch.mm(emb_norm, emb_norm.T) 
    sim_matrix = torch.exp(sim_matrix / tau) 

    topk_vals, topk_idx = torch.topk(sim_matrix, K+1, dim=1)
    topk_vals = topk_vals[:, 1:] 
    topk_idx = topk_idx[:, 1:]


    neighbors_preds = pred_probs[topk_idx] 
    weights = topk_vals / topk_vals.sum(dim=1, keepdim=True)
    weighted_neighbors_preds = (weights.unsqueeze(2) * neighbors_preds).sum(dim=1)
    qi = mu * pred_probs + (1 - mu) * weighted_neighbors_preds  
    print(f"Avg qi {qi.max(dim=1).values.mean().item()}")

    delta_c = delta_prime * (c / cmax)
    qi_max_class = qi.argmax(dim=1)

    clean_mask = (qi_max_class == original_labels)  
    max_qi_vals, _ = qi.max(dim=1)
    confident_mask = max_qi_vals > delta_c
    print(f"Updating {((confident_mask & (~clean_mask))*1.0).mean()} of the labels")
    updated_mask = clean_mask | confident_mask
    new_labels = torch.where(clean_mask, original_labels, qi_max_class)

    return updated_mask, new_labels, qi

def filter_dataset_with_label_correction(train_dataset, updated_mask, new_labels):
    filtered_dataset = []
    for i, sample in tqdm(enumerate(train_dataset)):
        if updated_mask[i].item():
            sample.y = new_labels[i].unsqueeze(0) 
            filtered_dataset.append(sample)
    return filtered_dataset

def add_signed_noise(h, gamma=0.01):
    raw_noise = torch.randn_like(h)
    normed_noise = F.normalize(raw_noise, p=2, dim=1)
    scaled_noise = gamma * normed_noise
    signed_noise = torch.abs(scaled_noise) * torch.sign(h)
    return h + signed_noise


def mixup_embeddings(embeddings, labels, alpha=0.1, beta=0.1):
    batch_size = embeddings.size(0)
    device = embeddings.device
    lam = torch.distributions.Beta(alpha, beta).sample([batch_size]).to(device).view(-1, 1)
    index = torch.randperm(batch_size).to(device)
    mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_embeddings, mixed_labels

def get_positive_mask(mixed_labels, threshold=0.05):
    diff = mixed_labels.unsqueeze(1) - mixed_labels.unsqueeze(0)
    l2_dist = torch.norm(diff, p=2, dim=2)
    pos_mask = (l2_dist < threshold).float()
    pos_mask.fill_diagonal_(0.0) 
    return pos_mask

def get_negatives(z):
    z_norm = F.normalize(z, p=2, dim=1)
    cosine_sim = torch.matmul(z_norm, z_norm.T) 
    mask = 1.0 - torch.eye(z.size(0), device=z.device)
    masked_sim = cosine_sim.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(masked_sim, dim=1)
    soft_negatives = torch.matmul(weights, z)
    return soft_negatives

class OmgContrastiveLoss(nn.Module):
    def __init__(self):
        super(OmgContrastiveLoss, self).__init__()

    def forward(self, pred, positive_mask, neg):
        pred = F.normalize(pred, dim=1)
        neg = F.normalize(neg, dim=1)

        dists = torch.norm(pred.unsqueeze(1) - pred.unsqueeze(0), p=2, dim=2)
        pos_sum = (dists * positive_mask).sum(dim=1) 
        pos_counts = torch.clamp(positive_mask.sum(dim=1), min=1.0)
        pos_avg = pos_sum / pos_counts

        neg_dists = torch.norm(pred - neg, p=2, dim=1)  

        # print(pos_avg.sum().item(), neg_dists.sum().item())
        loss_per_sample = pos_avg - neg_dists
        return loss_per_sample.mean()

class SupervisedSoftLabelLoss(nn.Module):
    def __init__(self):
        super(SupervisedSoftLabelLoss, self).__init__()

    def forward(self, y, y_pred):
        log_probs = F.log_softmax(y_pred, dim=1)
        loss = -(y*log_probs).sum(dim=1).mean() 
        return loss
    

def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    num_layer = 5
    emb_dim = 128
    drop_ratio = 0.5
    num_epochs = 15
    warm_up_epochs = 10
    batch_size = 32

    device = "cuda:0"
    num_class = 6

    test_path = args.test_path
    train_path = args.train_path

    # Prepare test dataset and loader
    test_dataset = GraphDataset(test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    embedding_model = GNN(gnn_type="gin", num_class=num_class, num_layer = num_layer, emb_dim = emb_dim, drop_ratio = drop_ratio, graph_pooling = "mean").to(device)
    embedding_projector = MLP([emb_dim, 32, 16]).to(device)
    classifier = MLP([emb_dim, num_class]).to(device)
    model = CompleteModel(embedding_model, classifier)

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


    if not train_path is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
        contrastive_optimizer = torch.optim.Adam(list(embedding_model.parameters()) + list(embedding_projector.parameters()) + list(classifier.parameters()), lr=0.0001, weight_decay=0.0005)

        contrastive_loss = OmgContrastiveLoss()
        supervised_label_loss = SupervisedSoftLabelLoss()

        criterion1 = GCELoss(q=0.7, num_classes=num_class)
        criterion2 = EntropyLoss()
        
        train_dataset = GraphDataset(train_path, transform=add_zeros)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(warm_up_epochs):
            model.train(); embedding_projector.train()
            total_loss = 0.0
            for data in tqdm(train_loader, desc="Iterating training graphs"):
                optimizer.zero_grad()
                data = data.to(device)
                logits = model(data)
                loss1, loss2 = criterion1(logits, data.y), criterion2(logits)
                # print(logits)
                # print(F.one_hot(logits.argmax(dim=1), num_classes=num_class).float().cpu().mean(dim=0), loss1, loss2)
                loss = loss1 - 5. * loss2
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            total_loss = total_loss / len(train_loader)
            model.eval();  embedding_projector.eval()
            p_dist = torch.zeros((num_class, ))
            with torch.no_grad():
                correct = 0
                total = 0
                for data in tqdm(train_loader, desc="Iterating eval graphs"):
                    data = data.to(device)
                    pred = model(data)
                    pred = pred.argmax(dim=1)
                    p_dist += F.one_hot(pred, num_classes=num_class).float().cpu().mean(dim=0)
                    correct += (pred == data.y).sum().item()
                    total += data.y.size(0)
                accuracy = correct / total

            print(f"Epoch {epoch + 1}/{warm_up_epochs}, Loss: {total_loss:.4f}, Train Acc: {accuracy:.4f}")
            print(f"Pred distribution: {(p_dist / len(train_loader)).tolist()}")

        checkpoint_file = os.path.join(checkpoints_folder, f"model_{test_dir_name}_warmup.pth")
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")


        beta = 1

        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []

        filtered_train_dataset = train_dataset

        c_max = 1
        for c in range(c_max):
            filtered_train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, shuffle=True)
            # embeddings, pred_probs, original_labels = compute_embeddings_and_preds(filtered_train_loader, embedding_model, classifier)
            # updated_mask, new_labels, _ = neighbor_aware_label_correction(embeddings, pred_probs, original_labels, c=c, cmax=c_max, K=10, mu=0.8)
            # print(f"Keeping {(updated_mask*1.).mean()} of previous data")
            # filtered_train_dataset = filter_dataset_with_label_correction(filtered_train_dataset, updated_mask, new_labels)
            # filtered_train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, shuffle=True)


            for epoch in range(num_epochs): 
                embedding_model.train();  embedding_projector.train(); classifier.train()
                total_loss = 0.0
                for data in tqdm(filtered_train_loader, desc="Iterating training graphs"):
                    contrastive_optimizer.zero_grad()
                    data = data.to(device)
                    label = F.one_hot(data.y, num_classes=num_class).float()
                    embeddings = embedding_model(data)
                    embeddings = add_signed_noise(embeddings)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    z, y = mixup_embeddings(embeddings, label)
                    positive_mask = get_positive_mask(y)
                    negatives = get_negatives(z)

                    emb_proj, neg_proj = embedding_projector(z), embedding_projector(negatives)
                    y_pred = classifier(z)
                    cl_loss, sup_loss = contrastive_loss(emb_proj, positive_mask, neg_proj), supervised_label_loss(y, y_pred)
                    loss = beta * cl_loss + sup_loss
                    loss.backward()
                    contrastive_optimizer.step()
                    total_loss += loss.item()
                
                embedding_model.eval();  classifier.eval(); embedding_projector.train()
                p_dist = torch.zeros((num_class, ))
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in tqdm(train_loader, desc="Iterating eval graphs"):
                        data = data.to(device)
                        output = embedding_model(data)
                        output = F.normalize(output, p=2, dim=1)
                        pred = classifier(output)
                        pred = pred.argmax(dim=1)
                        p_dist += F.one_hot(pred, num_classes=num_class).float().cpu().mean(dim=0)
                        correct += (pred == data.y).sum().item()
                        total += data.y.size(0)
                    accuracy = correct / total
                    
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Train Acc: {accuracy:.4f}")
                print(f"Pred distribution: {(p_dist / len(train_loader)).tolist()}")

                train_losses.append(total_loss)
                train_accuracies.append(accuracy)

                checkpoint_file = os.path.join(checkpoints_folder, f"model_{test_dir_name}_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_file)
                print(f"Checkpoint saved at {checkpoint_file}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Best model updated and saved at {checkpoint_path}")


        torch.save(model.state_dict(), "./develop_checkpoint/contrastive_trained_model.pth")
        torch.save(embedding_projector.state_dict(), "./develop_checkpoint/contrastive_trained_projector_model.pth")

    model.load_state_dict(torch.load(checkpoint_path))
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")

    args = parser.parse_args()
    main(args)


