import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
import anndata as ad

class AnnDataDataset(Dataset):
    def __init__(self, anndata):
        self.data = anndata

    def __len__(self):
        return self.data.n_obs

    def __getitem__(self, idx):
        return torch.tensor(self.data.X[idx], dtype=torch.float32)

class ExpressionData_And_Or(Dataset):
    def __init__(self, x_act, x_rep, y, t):
        if isinstance(x_act, np.ndarray):
            x_act = torch.from_numpy(x_act).type(torch.FloatTensor)
        if isinstance(x_rep, np.ndarray):
            x_rep = torch.from_numpy(x_rep).type(torch.FloatTensor)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).type(torch.FloatTensor).reshape(-1, 1)
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).type(torch.FloatTensor).reshape(-1, 1)

        # Not necessarily needed rn
        # assert x_act.size(0) == x_rep.size(0) == y.size(0) == t.size(0), "Size mismatch between time lengths"

        self.length = x_act.size(0) # arbitrary choice of x_act

        self.x_data_act = x_act
        self.x_data_rep = x_rep
        self.y_data = y
        self.t_data = t

    def __getitem__(self, index):
        return self.x_data_act[index], self.x_data_rep[index], self.y_data[index], self.t_data[index]

    def __len__(self):
        return self.length
    

