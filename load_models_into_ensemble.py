from src.models import EnsembleModel, VGAE
import torch

def load_old_models_into_ensemble():
    device = "cuda:0"
    save_as = "./checkpoints/model_D_loaded_ensemble.pth"
    # files = [
    #     "checkpoints_old/model_A_round_23_f1_0.7223_best.pth",
    #     "checkpoints_old/model_A_round_22_f1_0.7005_best.pth",
    #     "checkpoints_old/model_A_round_21_f1_0.7135_best.pth",
    #     "checkpoints_old/model_A_round_20_f1_0.7142_best.pth"
    # ]
    files = [
        "./checkpoints_old/model_D_round_9_f1_0.7486_best.pth",
        "./checkpoints_old/model_D_round_10_f1_0.7647_best.pth",
        "./checkpoints_old/model_D_round_23_f1_0.7719_best.pth"
    ]
    models = []
    weights = []
    for path in files:
        model = VGAE(in_channels=1, edge_attr_dim=7, hidden_dim=128, latent_dim=16, num_classes=6).to(device)
        f1 = float(path.split("_")[-2])
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model)
        weights.append(f1)

    ensemble = EnsembleModel(models, weights, device)
    torch.save(ensemble.state_dict(), save_as)

def merge_two_ensembles():
    device = "cuda:0"
    path1 = "./checkpoints/model_D_loaded_ensemble.pth"
    cnt1 = 3
    path2 = "./checkpoints/model_D_best.pth"
    cnt2 = 5

    save_to = "./checkpoints/model_D_merged_ensemble.pth"
    def get_model():
        return VGAE(in_channels=1, edge_attr_dim=7, hidden_dim=128, latent_dim=16, num_classes=6).to(device)

    model1 = EnsembleModel([get_model() for _ in range(cnt1)], [1 for _ in range(cnt1)], device)
    model1.load_state_dict(torch.load(path1, map_location=device))
    model2 = EnsembleModel([get_model() for _ in range(cnt2)], [1 for _ in range(cnt2)], device)
    model2.load_state_dict(torch.load(path2, map_location=device))


    combined_models = list(model1.models) + list(model2.models)
    print(len(combined_models))
    # Combine and normalize weights
    combined_weights = torch.cat([model1.weights, model2.weights])
    combined_weights = combined_weights / combined_weights.sum()

    # Create a new ensemble
    merged_ensemble = EnsembleModel(combined_models, combined_weights.tolist(), device)
    torch.save({
        "model_cnt": cnt1 + cnt2,
        "model_state_dict": merged_ensemble.state_dict()
    }, save_to)


def resave_ensemble_without_cnt():
    device = "cuda:0"
    path = "./checkpoints/model_A_best.pth"
    cnt = 5
    save_to = "./checkpoints/model_A_resaved_ensemble.pth"
    def get_model():
        return VGAE(in_channels=1, edge_attr_dim=7, hidden_dim=128, latent_dim=16, num_classes=6).to(device)

    model = EnsembleModel([get_model() for _ in range(cnt)], [1 for _ in range(cnt)], device)
    model.load_state_dict(torch.load(path, map_location=device))
    torch.save({
        "model_cnt": cnt,
        "model_state_dict": model.state_dict()
    }, save_to)

def reconstruct_lost_ensemble():
    s = "B"
    cnt = 9
    paths = [
        f"checkpoints/model_{s}_voter_{i}_best.pth" for i in range(cnt)
    ]
    
    device = "cuda:0"
    def get_model():
        return VGAE(in_channels=1, edge_attr_dim=7, hidden_dim=128, latent_dim=16, num_classes=6).to(device)
    
    models = []
    for path in paths:
        model = VGAE(in_channels=1, edge_attr_dim=7, hidden_dim=128, latent_dim=16, num_classes=6).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model)
    # weights = [0.8461, 0.8363, 0.8416, 0.8523, 0.8463, 0.8491, 0.8431, 0.8342, 0.8437]
    weights = [0.5563, 0.5725, 0.5484, 0.5592, 0.5382, 0.5318, 0.5916, 0.5405, 0.5634]
    ensemble = EnsembleModel(models, weights, device)
    torch.save({
        "model_cnt": cnt,
        "model_state_dict": ensemble.state_dict()
    }, f"checkpoints/model_{s}_best.pth")


if __name__ == "__main__":
    reconstruct_lost_ensemble()