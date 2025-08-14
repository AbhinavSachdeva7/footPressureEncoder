import scipy.io
import torch
import numpy as np
from glob import glob
import os.path as osp
import mat73

# all_subjects = osp.join('/scratch/data/PSUTMM100/PSU100/Modality_wise/Foot_Pressure','Subject1', 'Pressure_2.mat')
# print((all_subjects))

arr = []
sub = 10
end = 11

SUBJECT_TAKE_MAPPING = {
    # "1": range(1, 10),  # Process Subject 1, takes 1 through 9
    "2": range(1, 7),   # Process Subject 5, takes 1 through 8
    "3": range(1, 10),
    "4": range(1, 12),
    "5": range(1, 9),
    "6": range(1, 11),
    "7": range(1, 10),
    "8": range(1, 11),
    "9": range(1, 14),
    "10": range(1, 11),

    # 2,7 3,10  4,12  5,9  6,11   7,10  8,11  9,14  10, 11
    # Add other subjects and their take ranges here
}

# # Step 1: Load .mat file
def convert_mat_to_pt(sub, take_range):
    arr = []
    for i in take_range:
        x = osp.join('/scratch/data/PSUTMM100/PSU100/Modality_wise/Foot_Pressure',f'Subject{sub}', f'Pressure_{i}.mat')
        data = mat73.loadmat(x)  # Replace 'your_file.mat' with your actual file path

        # Step 2: Extract variable (replace 'variable_name' with your variable's name)
        # print(type(data))
        # print(data.keys())
        numpy_array = data['PRESSURE']

        # Step 3: Convert NumPy array to PyTorch tensor
        tensor = torch.from_numpy(numpy_array).int()

        # print("Tensor:", tensor)
        print("Tensor Shape:", tensor.shape)
        tensor1 = torch.transpose(tensor, 3, 2)
        tensor1 = tensor1.reshape(tensor.size(0), tensor.size(1), 1, 42)
        tensor1 = tensor1.squeeze(2)
        tensor1 = tensor1.reshape(tensor.size(0), 2520)
        # np.set_printoptions(threshold=np.inf, linewidth = 1000)
        # print(tensor1[781].numpy())
        print(tensor1.shape)
        arr.append(tensor1)
        # break
    #     # print(arr)
    #     # torch.save(tensor1, '/content/subj6_take2_pressure.pt')


        newTensor = torch.concat(arr, dim=0)
        print(newTensor.shape)
        torch.save(newTensor, f'/scratch/avs7793/work_done/poseembroider/new_model/src/data/pressure/pressure_subject{sub}_take{i}.pt')
        arr = []
        newTensor = None
        print(newTensor)


# torch.save(newTensor, 'all_subjects_pressure_96_files.pt')

for sub, take_range in SUBJECT_TAKE_MAPPING.items():
    convert_mat_to_pt(sub, take_range)