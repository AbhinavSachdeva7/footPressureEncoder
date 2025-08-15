import torch
from torch import nn


from model import VariationalAutoEncoder
from newModel import LargeVariationalAutoEncoder
from convEncoder import ConvolutionVariationalAutoEncoder
from data_utils import load_and_split_data







# DATA_PATH = '/scratch/avs7793/footPressureEncoder/footPressureEncoder/data/all_subjects_pressure.pt'
DATA_PATH = '/scratch/avs7793/footPressureEncoder/all_subjects_pressure_96_files_zeroed.pt'
TRAIN_RATIO = 0.8
BATCH_SIZE = 2048
SEED = 781


vae = VariationalAutoEncoder(
    input_dim=2520,
    h_dim=512,
    z_dim=32
)

vae1 = VariationalAutoEncoder(
    input_dim=2520,
    h_dim=512,
    z_dim=64
)

largeVAE = LargeVariationalAutoEncoder(
    input_dim = 2520,
    l_dim=3072,
    h_dim=1024,
    z_dim=64
)

convVAE = ConvolutionVariationalAutoEncoder(
    z_dim = 64
)

# CHECKPOINT_PATH = ''
SEED = 781

# model.eval()




def compute_average_mse(model, data_loader, device):
    
    """
    Compute the average per-sample MSE loss for a given data loader.

    Args:
        model (nn.Module): The trained model to evaluate.
        data_loader (DataLoader): PyTorch DataLoader for the dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Average per-sample MSE loss over the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch in data_loader:
            # print(batch.shape, batch[0].shape)
            # inputs_original = batch.to(device).unsqueeze(1)  # Move inputs to the same device as the model
            print(batch.shape)
            inputs_original = batch.to(device).view(-1,2520)
            
            inputs = torch.log1p(inputs_original)
            # print(inputs.shape)
            # break
            reconstructed, _, _ = model(inputs)  # Get model predictions (reconstructed outputs)
            reconstructed = torch.expm1(reconstructed)
            # Compute MSE loss: average squared error per element in the batch
            loss = nn.MSELoss(reduction='mean')(reconstructed, inputs_original)
            batch_size = inputs_original.numel()  # Number of samples in the batch
            total_loss += loss.item() * batch_size  # Accumulate total loss
            total_samples += batch_size  # Accumulate total number of samples

    # Calculate average per-sample MSE
    average_mse = total_loss / total_samples if total_samples > 0 else 0.0
    return average_mse



# CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/data_processing/model_epoch_201.pt'
# CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/dp_64_z/model_epoch_200.pt'
# CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/lg_start_z_64/model_epoch_200.pt'
# CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/convVAE/model_epoch_200.pt'
# CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/footPressureEncoder/checkpoints/model_200.pt'
# CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/dp_z64_ngc/model_epoch_100.pt'
# CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/convVAE_ngc/model_epoch_100.pt'
# CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/lg_z64_ngc/model_epoch_300.pt'
# CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/lg_z64_ngc/model_epoch_500_vhigh_fixed_b3.pt'
CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/large_z64_all_data_ngc/model_epoch_300.pt'

if __name__ == "__main__":

    device = torch.device('cuda')
    
    # model = vae.to(device)
    # model = vae1.to(device)
    model = largeVAE.to(device)
    # model = convVAE.to(device)
    
    
    
    train_loader, test_loader = load_and_split_data(file_path=DATA_PATH, train_ratio=TRAIN_RATIO, batch_size=BATCH_SIZE, seed=SEED)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    
    test_mse = compute_average_mse(model, test_loader, device)
    train_mse = compute_average_mse(model, train_loader, device)

    print(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
