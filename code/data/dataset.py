import torch
import torch.utils.data as data
import pickle

class DanceDataset(data.Dataset):
    def __init__(self, data_path):
        super(DanceDataset, self).__init__()
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)


        
    def __getitem__(self, index):
        cur_data = self.data[index]
        cur_data["coordinates"] = torch.tensor(cur_data["coordinates"])
        return cur_data

    def __len__(self):
        return len(self.data)
        
        
