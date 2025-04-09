# Import Modules
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.datasets as datasets
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset, random_split



# Load Datasets, 
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


dataset = train_data
print((dataset[2].shape))
print(len(dataset))


INPUT_DIM = 2520
H_DIM = 512
Z_DIM = 32
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
device = torch.device("cuda")

loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VariationalAutoEncoder(input_dim=INPUT_DIM, h_dim=H_DIM, z_dim=Z_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss(reduction="sum")

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(loader))
    total_loss = 0
    for (i, x) in loop:
        x = x.to(device).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, log_var = model(x)
        # print(x_reconstructed.shape, mu.shape, log_var.shape)

        reconstruction_loss = loss_fn(x_reconstructed, x)
        # kl_div = -torch.sum(1 + torch.log(log_var.pow(2)) - mu.pow(2) - log_var.pow(2))
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader.dataset)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    if (epoch%100==0):
        torch.save(model.state_dict(), f"model_{epoch}.pt")

