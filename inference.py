# Import Modules
import torch
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader # Keep DataLoader

# Local imports
from model import VariationalAutoEncoder
from newModel import LargeVariationalAutoEncoder
from convEncoder import ConvolutionVariationalAutoEncoder
from data_utils import load_and_split_data # Import data loading utility

# --- Configuration Constants ---
DATA_PATH = '/scratch/avs7793/footPressureEncoder/footPressureEncoder/data/all_subjects_pressure.pt'
CHECKPOINT_PATH = '/scratch/avs7793/footPressureEncoder/checkpoints/lg_z64_ngc/model_epoch_505_vhigh_fixed_b3.pt' # Specify the exact checkpoint

# Data handling constants (should match train.py for consistency)
TRAIN_RATIO = 0.8
SPLIT_SEED = 781 # Use the same seed as training to get the correct test set
BATCH_SIZE = 64 # Batch size for loading test data (can be different from training)

# Model constants (must match the loaded checkpoint)
H_DIM = 512
Z_DIM = 64
INPUT_DIM = 2520

# Inference constants
SINGLE_SAMPLE_IDX = 35700 # Index for the single specific sample reconstruction
NUM_TEST_SAMPLES_TO_VISUALIZE = 20 # Number of test samples to plot
OUTPUT_DIR = '/scratch/avs7793/footPressureEncoder/examples/examples_lg_z64_ngc'
FORCE_CPU = True # Set to False to try using GPU if available
IMAGE_SHAPE = (60, 42) # Define image shape for reshaping
# --- End Configuration ---

def plot_reconstruction(original, reconstructed, sample_idx_str, save_path, mse=None):
    """Plots the original and reconstructed images side-by-side, optionally displaying MSE."""
    original_image = original.view(*IMAGE_SHAPE).cpu().numpy()
    reconstructed_image = reconstructed.view(*IMAGE_SHAPE).cpu().numpy()

    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='viridis')
    plt.title(f"Original Sample #{sample_idx_str}")
    plt.colorbar()
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='viridis')
    title = "Reconstructed Sample"
    if mse is not None:
        title += f"\n(MSE: {mse:.4f})"
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.suptitle("VAE Reconstruction Quality")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Reconstruction plot saved to: {save_path}")
    # plt.show() # Optionally uncomment to display plot interactively
    plt.close() # Close the figure to free memory

def evaluate_and_visualize_test_samples(model, test_loader, device, output_dir, input_dim, num_samples=10):
    """Runs inference on test samples, calculates MSE, prints it, and saves plots."""
    print(f"\nEvaluating and visualizing {num_samples} test samples...")
    model.eval() # Ensure model is in evaluation mode
    mse_fn = nn.MSELoss(reduction='mean') # Use mean MSE for per-sample reporting
    samples_saved = 0

    with torch.no_grad():
        for batch_idx, x_original_batch in enumerate(test_loader):
            if samples_saved >= num_samples:
                break

            # Move to device and reshape
            x_original_batch = x_original_batch.to(device).view(x_original_batch.shape[0], input_dim)

            # for convolutional vae
            # x_original_batch = x_original_batch.to(device).unsqueeze(1)

            # Apply log1p preprocessing (as done in training)
            x_processed = torch.log1p(x_original_batch)

            # Pass processed data to the model
            x_reconstructed_processed, _, _ = model(x_processed)

            # Apply expm1 transformation (inverse of preprocessing)
            x_reconstructed = torch.expm1(x_reconstructed_processed)

            # Process samples within the batch
            for i in range(x_original_batch.shape[0]):
                if samples_saved >= num_samples:
                    break

                original_sample = x_original_batch[i]
                reconstructed_sample = x_reconstructed[i]

                # Calculate MSE for this sample
                mse = mse_fn(reconstructed_sample, original_sample).item()
                print(f"Test Sample {samples_saved + 1} MSE: {mse:.4f}")

                # Plotting
                plot_filename = f"test_reconstruction_sample_{samples_saved + 1}_mse_{mse:.4f}.png"
                output_path = os.path.join(output_dir, plot_filename)
                plot_reconstruction(
                    original=original_sample,
                    reconstructed=reconstructed_sample,
                    sample_idx_str=f"Test {samples_saved + 1}",
                    save_path=output_path,
                    mse=mse
                )
                samples_saved += 1

    print(f"Finished visualizing {samples_saved} test samples.")

# --- Main Script Logic ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    # Use data_utils to get the *exact same* train/test split as used in training
    # We only need the test_loader and input_dim here.
    print("Loading data and getting test split...")
    _, test_loader = load_and_split_data(
        file_path=DATA_PATH,
        train_ratio=TRAIN_RATIO,
        batch_size=BATCH_SIZE,
        seed=SPLIT_SEED # Use same seed as training!
    )

    if test_loader is None:
        print("Error: Could not obtain test loader. Check TRAIN_RATIO.")
        exit()

    # --- Initialize Model ---
    # model = VariationalAutoEncoder(
    #     input_dim=2520, # Use inferred input_dim
    #     h_dim=512,
    #     z_dim=32
    # ).to(device)


    # model = VariationalAutoEncoder(
    #     input_dim=2520, # Use inferred input_dim
    #     h_dim=512,
    #     z_dim=64
    # ).to(device)

    model = LargeVariationalAutoEncoder(
        input_dim=2520,
        l_dim = 3072,
        h_dim = 1024,
        z_dim = 64
    ).to(device)

    # model = ConvolutionVariationalAutoEncoder(
    #     z_dim = 64
    # ).to(device)

    # --- Load Checkpoint ---
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint path '{CHECKPOINT_PATH}' not provided or not found. Exiting.")
        exit()

    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    try:
        # Load state dict (map_location ensures compatibility if trained on GPU and inferring on CPU)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.eval() # Set model to evaluation mode
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Exiting.")
        exit()

    # --- Evaluate and Visualize Test Samples ---
    # evaluate_and_visualize_test_samples(
    #     model=model,
    #     test_loader=test_loader,
    #     device=device,
    #     output_dir=OUTPUT_DIR,
    #     input_dim=INPUT_DIM,
    #     num_samples=NUM_TEST_SAMPLES_TO_VISUALIZE
    # )

    # # --- Original Single Sample Inference (Optional) ---
    # # Load the full dataset again just to get the specific single sample
    # # This is inefficient but keeps the logic separate for demonstration
    # print(f"\nLoading full dataset again for single sample index {SINGLE_SAMPLE_IDX}...")
    # try:
    #     full_dataset = torch.load(DATA_PATH).float()
    #     if SINGLE_SAMPLE_IDX >= len(full_dataset):
    #          print(f"Error: Single sample index {SINGLE_SAMPLE_IDX} is out of bounds (Dataset size: {len(full_dataset)}). Using index 0.")
    #          single_sample_idx_to_use = 0
    #     else:
    #          single_sample_idx_to_use = SINGLE_SAMPLE_IDX

    #     original_x_single = full_dataset[single_sample_idx_to_use].to(device)
    #     original_x_single_batch = original_x_single.unsqueeze(0) # Add batch dimension
    #     original_x_single_batch = original_x_single_batch.view(1, INPUT_DIM) # Ensure flattened

    #     # Perform inference for the single sample
    #     print(f"Running inference for single sample index: {single_sample_idx_to_use}")
    #     with torch.no_grad():
    #         # Apply log1p preprocessing
    #         x_processed_single = torch.log1p(original_x_single_batch)
    #         reconstructed_x_processed_single, _, _ = model(x_processed_single)
    #         # Apply expm1 transformation
    #         reconstructed_x_single = torch.expm1(reconstructed_x_processed_single)

    #     # Plotting the single sample
    #     plot_filename_single = f"single_reconstruction_sample_{single_sample_idx_to_use}.png"
    #     output_path_single = os.path.join(OUTPUT_DIR, plot_filename_single)
    #     # Calculate MSE for the single sample
    #     mse_fn_single = nn.MSELoss(reduction='mean')
    #     mse_single = mse_fn_single(reconstructed_x_single.squeeze(0), original_x_single_batch.squeeze(0)).item()
    #     print(f"Single Sample {single_sample_idx_to_use} MSE: {mse_single:.4f}")

    #     plot_reconstruction(
    #         original=original_x_single,
    #         reconstructed=reconstructed_x_single.squeeze(0),
    #         sample_idx_str=f"Single {single_sample_idx_to_use}",
    #         save_path=output_path_single,
    #         mse=mse_single
    #         )
    # except FileNotFoundError:
    #     print(f"Error: Data file not found at {DATA_PATH} when trying to load for single sample.")
    # except Exception as e:
    #     print(f"Error during single sample processing: {e}")

# --- Optional: Generate samples from random Z --- (kept original commented code structure)
try:
    with torch.no_grad():
        num_generated_samples = 2
        torch.manual_seed(781)
        random_z = torch.randn(num_generated_samples, 64).to(device)
        generated_x_flat = model.decode(random_z)
        generated_x_flat = torch.expm1(generated_x_flat)
    generated_images = generated_x_flat.view(num_generated_samples, *IMAGE_SHAPE).cpu().numpy()
    # generated_images = generated_x_flat.squeeze(1).cpu()
    # print(generated_images.shape)
    plt.figure(figsize=(num_generated_samples * 3, 4))
    for i in range(num_generated_samples):
        plt.subplot(1, num_generated_samples, i + 1)
        plt.imshow(generated_images[i], cmap='viridis')
        plt.title(f"Generated Sample {i+1}")
        plt.axis('off')
    plt.suptitle("VAE Generated Samples from Random Latent Vectors (z)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    gen_save_path = os.path.join(OUTPUT_DIR, "generated_samples_only2.png")
    os.makedirs(os.path.dirname(gen_save_path), exist_ok=True)
    plt.savefig(gen_save_path)
    print(f"Generated samples plot saved to: {gen_save_path}")
    # plt.show()
    plt.close()
except Exception as e:
    print(f"Error during generation visualization: {e}")

print("\nInference complete.")
