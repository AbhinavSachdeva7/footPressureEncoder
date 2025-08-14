# footPressureEncoder

This project implements a Variational Autoencoder (VAE) to learn a compressed representation of foot pressure data. The VAE can be used for various tasks, such as anomaly detection, data generation, and feature extraction.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Visualization](#visualization)

## Directory Structure

```
footPressureEncoder/
├── checkpoints/              # Stores saved model checkpoints
├── data/                     # Stores processed data in .pt format
├── examples/                 # Stores output visualizations
├── .gitignore                # Specifies files to be ignored by Git
├── convEncoder.py            # Convolutional VAE model definition
├── data_utils.py             # Utilities for loading and splitting data
├── inference.py              # Script for running inference and visualization
├── mat_to_pt.py              # Script for converting .mat files to .pt
├── model.py                  # Standard VAE model definition
├── newModel.py               # Large VAE model definition
├── README.md                 # This file
├── train.py                  # Script for training the VAE
└── validate.py               # Script for validating the model's performance
```

## Requirements
A lot of the requirements present are not really being used but this encoder is connected to the retrieval model built based on this and the requirements make sure that both of these models can run.

### Libraries
You can install the required Python libraries using pip. For further information the packages present in the conda environment have also been uploaded in the repo. 

```bash
pip install -r requirements.txt
```

### Data
The model expects foot pressure data in the `.mat` file format. Specifically, the project was designed to work with the PSU-TMM-100 dataset, but it can be adapted for other datasets with a similar structure. The raw data should be stored in a directory accessible by the `mat_to_pt.py` script. The path to this directory is hardcoded in the script, so you may need to modify it to match your setup.

## Data Preparation

The raw `.mat` files need to be converted to PyTorch tensors (`.pt` files) before they can be used for training. This is done using the `mat_to_pt.py` script.

1.  **Configure `mat_to_pt.py`**:
    *   Open `footPressureEncoder/mat_to_pt.py`.
    *   In the `SUBJECT_TAKE_MAPPING` dictionary, specify the subjects and takes you want to process.
    *   Update the path in the `osp.join` function to point to the location of your raw `.mat` files.
    *   Update the output path in the `torch.save` function to specify where the processed `.pt` files should be saved.

2.  **Run the script**:

    ```bash
    python footPressureEncoder/mat_to_pt.py
    ```

This will generate a `.pt` file for each subject and take combination specified in the `SUBJECT_TAKE_MAPPING` dictionary. These files will be saved in the directory you specified in the script.

## Training

The `train.py` script is used to train the VAE model.

1.  **Configure `train.py`**:
    *   Open `footPressureEncoder/train.py`.
    *   Set the `DATA_PATH` variable to the path of your processed `.pt` file.
    *   Adjust the hyperparameters, such as `TRAIN_RATIO`, `NUM_EPOCHS`, `BATCH_SIZE`, and `LEARNING_RATE`, as needed.
    *   If you want to resume training from a checkpoint, set the `CHECKPOINT_LOAD_PATH` to the path of the checkpoint file.
    *   Set the `CHECKPOINT_DIR` to the directory where you want to save new checkpoints.

2.  **Run the script**:

    ```bash
    python footPressureEncoder/train.py
    ```

The script will train the model and save checkpoints periodically to the directory specified by `CHECKPOINT_DIR`.

## Testing

The `validate.py` script is used to evaluate the performance of a trained model on the test set.

1.  **Configure `validate.py`**:
    *   Open `footPressureEncoder/validate.py`.
    *   Set the `DATA_PATH` variable to the path of your processed `.pt` file.
    *   Set the `CHECKPOINT_PATH` to the path of the trained model checkpoint you want to evaluate.

2.  **Run the script**:

    ```bash
    python footPressureEncoder/validate.py
    ```

The script will load the specified checkpoint, compute the mean squared error (MSE) on the training and test sets, and print the results.

## Visualization

The `inference.py` script is used to visualize the reconstructions produced by the trained model.

1.  **Configure `inference.py`**:
    *   Open `footPressureEncoder/inference.py`.
    *   Set the `DATA_PATH` variable to the path of your processed `.pt` file.
    *   Set the `CHECKPOINT_PATH` to the path of the trained model checkpoint you want to use for visualization.
    *   Set the `OUTPUT_DIR` to the directory where you want to save the visualization plots.
    *   You can also adjust other parameters, such as `NUM_TEST_SAMPLES_TO_VISUALIZE` and `SINGLE_SAMPLE_IDX`, to control the visualization process.

2.  **Run the script**:

    ```bash
    python footPressureEncoder/inference.py
    ```

The script will generate and save several plots in the specified output directory, including:
*   Side-by-side comparisons of original and reconstructed test samples.
*   Reconstruction of a single, specific sample.
*   Samples generated from random latent vectors.