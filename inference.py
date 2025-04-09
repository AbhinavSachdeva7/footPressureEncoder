# Import Modules
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.datasets as datasets
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt



# Load Datasets
file_path = '/scratch/avs7793/footPressureEncoder/data/all_subjects_pressure.pt'

load_data = torch.load(file_path).float()
train_ratio = 0.8
train_size = int(train_ratio * len(load_data)) # Number of samples = 116952
test_size = len(load_data) - train_size # No. of samples = 29231, close to 
train_data, test_data = random_split(load_data, [train_size, test_size])


# VAE Model


class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_dim, h_dim=200, z_dim=32):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2imd = nn.Linear(h_dim, h_dim)
        self.imd_2mu = nn.Linear(h_dim, z_dim)
        self.imd_2log_var = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2imd = nn.Linear(z_dim, h_dim)
        self.imd_2hid = nn.Linear(h_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        
        h = self.relu(self.img_2hid(x))
        h = self.relu(self.hid_2imd(h))
        mu, log_var = self.imd_2mu(h), self.imd_2log_var(h)
        return mu, log_var

    def decode(self,z):
        h = self.relu(self.z_2imd(z))
        # h = self.
        h = self.relu(self.imd_2hid(h))

        return self.hid_2img(h)

    def forward(self,x):
        # print("f1",x.shape)
        mu, log_var = self.encode(x)
        std = torch.exp(0.5*log_var)
       
        epsilon = torch.randn_like(std)
        z_reparametrized = mu + std*epsilon
        x_reconstructed = self.decode(z_reparametrized)

        return x_reconstructed, mu, log_var


# Training Loop


dataset = load_data
print((dataset[2].shape))
print(len(dataset))



# Saved checkpoint
checkpoint = True
checkpoint_file = '/scratch/avs7793/footPressureEncoder/checkpoints/model_200.pt'



INPUT_DIM = 2520
H_DIM = 512
Z_DIM = 32
NUM_EPOCHS = 101
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
device = torch.device("cpu")

loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VariationalAutoEncoder(input_dim=INPUT_DIM, h_dim=H_DIM, z_dim=Z_DIM).to(device)

if checkpoint : 
    model.load_state_dict(torch.load(checkpoint_file))
    model.eval()
    model.to(device)


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss(reduction="sum")



sample_idx = 28700 # Index of the sample to visualize
original_x = dataset[sample_idx].to(device) # Get single sample, move to device
original_x_batch = original_x.unsqueeze(0) # Add batch dimension
original_x_batch = original_x_batch.view(1, INPUT_DIM) # Ensure flattened
print(original_x_batch.shape)
with torch.no_grad(): # Disable gradient calculations
    reconstructed_x_flat, _, _ = model(original_x_batch)
# ------------------------------------------------


# Reshape for plotting
original_image = original_x.view(60, 42).cpu().numpy()
reconstructed_image = reconstructed_x_flat.view(60, 42).cpu().numpy()


# Plot Original vs Reconstructed
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='viridis') # Use 'gray' for grayscale, 'viridis' or others for heatmap/pressure
plt.title(f"Original Sample #{sample_idx}")
plt.colorbar()
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='viridis')
plt.title("Reconstructed Sample")
plt.colorbar()
plt.axis('off')
plt.suptitle("VAE Reconstruction Quality")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()
plt.savefig(f"original_{sample_idx}_reconstructed_0.png")




# try:
#     with torch.no_grad():
#         # Sample a random latent vector z from the prior (standard Gaussian)
#         # Generate a few samples (e.g., 4) to see variation
#         num_generated_samples = 4
#         random_z = torch.randn(num_generated_samples, Z_DIM).to(device)
#         print(f"Generated random z shape: {random_z.shape}")

#         # Decode z to generate new samples
#         generated_x_flat = model.decode(random_z)

#         # --- Optional: Apply Sigmoid for Visualization ---
#         # generated_x_flat = torch.sigmoid(generated_x_flat)
#         # ------------------------------------------------

#     # Reshape generated samples for plotting
#     generated_images = generated_x_flat.view(num_generated_samples, 60, 42).cpu().numpy()

#     # Plot Generated Samples
#     plt.figure(figsize=(num_generated_samples * 3, 4))
#     for i in range(num_generated_samples):
#         plt.subplot(1, num_generated_samples, i + 1)
#         plt.imshow(generated_images[i], cmap='viridis')
#         plt.title(f"Generated Sample {i+1}")
#         plt.axis('off')

#     plt.suptitle("VAE Generated Samples from Random Latent Vectors (z)")
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig('plot555.png')
#     plt.show()
# except Exception as e:
#     print(f"Error during generation visualization: {e}")

# print("\nVisualization complete.")
