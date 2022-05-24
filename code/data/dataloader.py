from torch.data import DataLoader
from dataset import DanceDataset


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    batch_dict = {"coordinate": [], "label":[],"fname":[], "frames":[]}
    coordinates = []
   
    for data in batch:
        for k, v in data.items():
            if k == "coordinate":
                batch_dict["length"].append(len(v))
                coordinates.append(v)
            else:
                batch_dict[k].append(v)
                

    batch_dict["label"] = torch.tensor(batch_dict["label"])
                
    coordinates = torch.nn.utils.rnn.pad_sequence(coordinates)
    batch_dict["coordinate"] = coordinates
 
    return batch_dict

def DanceDataLoader(data_path, batch_size):
    dataset = DanceDataset(data_path=data_path)
    dataloader = DataLoader(dataset ,shuffle=True, collate_fn = collate_fn_padd)
    
    return dataloader
    

