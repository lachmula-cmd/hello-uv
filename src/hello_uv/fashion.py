from torch.utils.data import Dataset, DataLoader,dataset
import torch
from torchvision import datasets,transforms


class ArrayDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32) 
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,i):
        return self.x[i], self.y[i]

def get_fashion(train=True, root="data"):
        return datasets.FashionMNIST(root=root, train=train, download=True, transform=transforms.ToTensor())

def make_loader(dataset, batch_size, shuffle):
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

def class_counts(loader):
        total = torch.zeros(10, dtype=torch.int64)

        for _,y in loader:
            total += torch.bincount(y, minlength=10)

        return total.tolist()