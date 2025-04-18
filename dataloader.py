####################################################################################################################
# % Code for the paper:
# % A Holistic Cybersecurity Framework against False Data Injection Attacks in Smart Water Distribution Systems Employing Auto-Encoders
# % Published in ASCE-EWRI 2024 proceedings and won Second best paper award in graduate paper competition  
# % By Nazia Raza; Farrah Moazeni, PhD
# % Lehigh University, nar522@lehigh.edu, moazeni@lehigh.edu
####################################################################################################################

import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np

class DatasetPrep(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

class LoadDataset:
    def __init__(self, scaled_train, scaled_test, scaled_val):
        self.scaled_train = scaled_train
        self.scaled_test = scaled_test
        self.scaled_val = scaled_val

    def load_data(self):
        X1 = torch.tensor(self.scaled_train.values, dtype=torch.float32)
        X2 = torch.tensor(self.scaled_test.values, dtype=torch.float32)
        X3 = torch.tensor(self.scaled_val.values, dtype=torch.float32)
        train_dataset = DatasetPrep(X1)
        test_dataset = DatasetPrep(X2) 
        val_dataset = DatasetPrep(X3) 

        return train_dataset, test_dataset, val_dataset
    
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
