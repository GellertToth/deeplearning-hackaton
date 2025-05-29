from src.models import EnsembleModel, VGAE
import torch

def main():
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

if __name__ == "__main__":
    main()