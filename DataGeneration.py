from torch.utils.data import Dataset
import pandas as pd
import os
import torch
# Create a custom Dataset


class DataGeneration(Dataset):
    def __init__(self, DIR_name, train=True):
        self.DIR_name = DIR_name
        self.train = train
        try:
            self.data = pd.read_csv(os.path.join(
                DIR_name, f"{'train' if train else 'test'}.csv"))
        except:
            raise FileNotFoundError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = (self.data.iloc[idx, :-1].values).reshape(64,
                                                        64).astype('float32')
        class_id = self.data.iloc[idx, -1]
        img_tensor = torch.from_numpy(img)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id
