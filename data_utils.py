import torch
from torch.utils.data import DataLoader, random_split

def load_and_split_data(file_path, train_ratio=0.8, batch_size=64, shuffle_train=True, seed=None):
    """Loads data, splits into train/test, creates DataLoaders, optionally using a fixed seed for splitting."""
    print(f"Loading data from: {file_path}")
    load_data = torch.load(file_path).float()
    print(f"Total samples loaded: {len(load_data)}")
    print(f"Sample shape: {load_data[0].shape}")

    if train_ratio < 1.0:
        train_size = int(train_ratio * len(load_data))
        test_size = len(load_data) - train_size
        print(f"Splitting data: Train ({train_size}), Test ({test_size})")

        # Set the seed *before* splitting if provided
        if seed is not None:
            print(f"Using fixed seed for splitting: {seed}")
            torch.manual_seed(seed)

        train_data, test_data = random_split(load_data, [train_size, test_size])
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    else:
        print("Using entire dataset for training.")
        train_data = load_data
        test_loader = None # No test set if using all data for training

    # Note: Seeding DataLoader shuffling requires a generator object passed to DataLoader
    # If consistent shuffling *within* epochs is also needed, further changes are required.
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle_train)

    # input_dim = load_data[0].nelement() # Calculate input dimension from data
    # print(f"Inferred input dimension: {input_dim}")

    return train_loader, test_loader

def get_full_dataset(file_path):
    """Loads the full dataset without splitting."""
    print(f"Loading full dataset from: {file_path}")
    load_data = torch.load(file_path).float()
    print(f"Total samples loaded: {len(load_data)}")
    print(f"Sample shape: {load_data[0].shape}")
    input_dim = load_data[0].nelement()
    print(f"Inferred input dimension: {input_dim}")
    return load_data, input_dim 

def process_data(data, batch_size, shuffle):

    """Loads data, splits into train/test, creates DataLoaders, optionally using a fixed seed for splitting, & processes the data by applying log(1+x)"""
    print(f"Loading data from: {file_path}")
    load_data = torch.load(file_path).float()
    print(f"Total samples loaded: {len(load_data)}")
    print(f"Sample shape: {load_data[0].shape}")

    if train_ratio < 1.0:
        train_size = int(train_ratio * len(load_data))
        test_size = len(load_data) - train_size
        print(f"Splitting data: Train ({train_size}), Test ({test_size})")

        # Set the seed *before* splitting if provided
        if seed is not None:
            print(f"Using fixed seed for splitting: {seed}")
            torch.manual_seed(seed)

        train_data, test_data = random_split(load_data, [train_size, test_size])
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    else:
        print("Using entire dataset for training.")
        train_data = load_data
        test_loader = None # No test set if using all data for training
    
    data = torch.log1p(data)
    loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle_train)
    return loader
    
    
    