# Import Modules
import torch
import torch.nn.functional as F
from torch import nn, optim
# Removed torchvision datasets/transforms/utils as they weren't used for this data
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

# Local imports
from model import VariationalAutoEncoder
from newModel import LargeVariationalAutoEncoder
from convEncoder import ConvolutionVariationalAutoEncoder
from data_utils import load_and_split_data
from validate import compute_average_mse

# --- Configuration Constants ---
DATA_PATH = '/scratch/avs7793/footPressureEncoder/all_subjects_pressure_96_files_zeroed.pt'
TRAIN_RATIO = 0.8
OVERFITTING_MULTIPLE = 1.6
INPUT_DIM = 2520
LH_DIM = 3072
H_DIM = 1024 # 1024 for LargeVAE
Z_DIM = 64
NUM_EPOCHS = 200 # Example: Run for 201 epochs
BATCH_SIZE = 2048
LEARNING_RATE = 1e-7
FORCE_CPU = False # Set to True to force CPU usage
# CHECKPOINT_LOAD_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/large_z64_all_data_ngc/model_epoch_300.pt'
# CHECKPOINT_LOAD_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/lg_z64_ngc/model_epoch_400.pt'
CHECKPOINT_LOAD_PATH = None
CHECKPOINT_DIR = '/scratch/avs7793/footPressureEncoder/checkpoints/lg_start_z_64/'
CHECKPOINT_SAVE_FREQ = 100 # Save every 100 epochs
SEED = 781
# --- End Configuration ---

def get_alpha_beta(kl_loss=0, recon_loss=1):
    # print("using function")
    ratio1 = recon_loss/kl_loss
    # print(ratio1)
    if ratio1 >= 80:
        return 1, 1
    elif ratio1>=60:
        return ratio1.item()*0.8, 1
    elif ratio1 >= 20:
        return ratio1.item()*1, 0.9
    elif ratio1 <= 10:
        return 1.0, 1.0
    # elif ratio1 >= 5:
    #     return 2.0
    else:
        return 1.0,1.0
    # print(ratio1.item())
    # return ratio1.item()



def train_epoch(model, loader, optimizer, loss_fn, device, input_dim, epoch, num_epochs):
    """Runs a single training epoch with log1p input and expm1 output transforms for loss."""
    loop = tqdm(enumerate(loader), total=len(loader), leave=False, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    total_loss = 0
    model.train()
    total_kl_div = 0
    total_reconstruction_loss = 0
    # try:
    #     beta = get_beta(avg_kl_loss)
    #     print("using the function")
    # except:
    #     beta = 10

    for (i, x_original) in loop:
        # Move to device and reshape
        x_original = x_original.to(device).view(x_original.shape[0], input_dim)
        
        # for conv VAE, adding channels
        # x_original = x_original.to(device).unsqueeze(1)
        
        # Apply log1p preprocessing
        
        optimizer.zero_grad()

        x_processed = torch.log1p(x_original)

        # Pass processed data to the model
        x_reconstructed_processed, mu, log_var = model(x_processed)

        # Apply expm1 transformation *before* loss calculation
        x_reconstructed = torch.expm1(x_reconstructed_processed)
    

        # Calculate reconstruction loss against the original input
        reconstruction_loss = loss_fn(x_reconstructed, x_original)
        # print(log_var.min(), log_var.max(), reconstruction_loss.min(), reconstruction_loss.max())
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = (reconstruction_loss) + (kl_div)
        
        loss.backward()
        # Optional: Gradient clipping (uncomment if needed)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        total_kl_div += kl_div
        total_reconstruction_loss += reconstruction_loss
        loop.set_postfix(loss=loss.item() / x_original.shape[0]) # Show loss per sample
        

    avg_loss = total_loss / len(loader.dataset)
    avg_kl_loss, avg_reconstruction_loss = total_kl_div/ len(loader.dataset), total_reconstruction_loss/ len(loader.dataset) 
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} Avg KL Loss: {avg_kl_loss:.4f} Avg Reconstruction Loss: {avg_reconstruction_loss:.4f}")
    return avg_loss, avg_kl_loss, avg_reconstruction_loss

def train(model, train_loader, test_loader, optimizer, loss_fn, device, input_dim, num_epochs, checkpoint_dir, checkpoint_save_freq):
    """Runs the main training loop for the specified number of epochs."""
    print("Starting training...")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.75)
    # beta = 10.0
    # alpha = 0.1
    for epoch in range(num_epochs):
        train_loss, avg_kl_loss, avg_recon_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, input_dim, epoch+300, num_epochs+300)
        # beta, alpha = get_alpha_beta(avg_kl_loss, avg_recon_loss)
        scheduler.step()
        
        if (epoch+1)%50 == 0:
            train_mse = compute_average_mse(model,train_loader, device=device)
            test_mse = compute_average_mse(model,test_loader, device=device)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f} Learning Rate: {current_lr:.6f}")
            if test_mse >= (OVERFITTING_MULTIPLE*train_mse):
                print("Risk of Overfitting")
                checkpoint_save_path = os.path.join(checkpoint_dir, f"model_epoch_{300+epoch+1}.pt")
                torch.save(model.state_dict(), checkpoint_save_path)
                print(f"Checkpoint saved to {checkpoint_save_path}")
                break

        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_save_freq == 0 or epoch == num_epochs - 1:
            # Ensure checkpoint directory exists
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_save_path = os.path.join(checkpoint_dir, f"model_epoch_{300+epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_save_path)
            print(f"Checkpoint saved to {checkpoint_save_path}")
    print("Training finished.")

# --- Main Script Logic ---
if __name__ == "__main__":
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = load_and_split_data(
        file_path=DATA_PATH,
        train_ratio=TRAIN_RATIO,
        batch_size=BATCH_SIZE,
        seed = SEED
    )

    # vae = VariationalAutoEncoder(
    #     input_dim=2520,
    #     h_dim=512,
    #     z_dim=32
    # )

    # vae1 = VariationalAutoEncoder(
    #     input_dim=2520,
    #     h_dim=512,
    #     z_dim=64
    # )

    largeVAE = LargeVariationalAutoEncoder(
        input_dim = 2520,
        l_dim=3072,
        h_dim=1024,
        z_dim=64
    )

    # convVAE = ConvolutionVariationalAutoEncoder(
    #     z_dim = 64
    # )

    # model = vae.to(device)
    # model = vae1.to(device)
    model = largeVAE.to(device)
    # model = convVAE.to(device)


    # Load checkpoint if specified
    if CHECKPOINT_LOAD_PATH:
        print(f"Loading checkpoint from: {CHECKPOINT_LOAD_PATH}")
        try:
            model.load_state_dict(torch.load(CHECKPOINT_LOAD_PATH, map_location=device))
            print("Checkpoint loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: Checkpoint file not found at {CHECKPOINT_LOAD_PATH}. Starting from scratch.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_fn = nn.MSELoss(reduction="sum") # Keep original MSE loss

    # --- Trigger Training ---
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        input_dim=INPUT_DIM,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_save_freq=CHECKPOINT_SAVE_FREQ
    )

