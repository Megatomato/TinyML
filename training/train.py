import torch, torchvision 
import torch.nn as nn 
import torch.nn.functional as F
import random
import numpy as np
from typing import Optional, Sequence, Tuple

def set_device(): 
    if torch.cuda.is_available(): 
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        return torch.device("mps")
    else: 
        return torch.device("cpu") 
    
DEVICE = set_device() 


def transform_data(): 
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    # need to do random transforms here?

def transform_train(): 
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(28, padding=2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

def transform_eval(): 
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_datasets(data_root: str = "data", download: bool = True): 

    classes = list(range(7))

    train_full = torchvision.datasets.MNIST(
        root=data_root,
        train=True,
        download=download,
        transform=transform_train(),
        target_transform=None,
    )
    test_full = torchvision.datasets.MNIST(
        root=data_root,
        train=False,
        download=download,
        transform=transform_eval(),
        target_transform=None,
    )

    keep_tensor = torch.tensor(classes)
    train_mask = torch.isin(train_full.targets, keep_tensor)
    test_mask = torch.isin(test_full.targets, keep_tensor)
    train_indices = torch.nonzero(train_mask, as_tuple=False).squeeze(1).tolist()
    test_indices = torch.nonzero(test_mask, as_tuple=False).squeeze(1).tolist()

    train_data = torch.utils.data.Subset(train_full, train_indices)
    test_data = torch.utils.data.Subset(test_full, test_indices)
    
    return train_data, test_data

def make_loaders(train_data, test_data): 
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    return train_loader, test_loader 

class Tiny(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)    
        self.conv2 = nn.Conv2d(6, 16, 5)     
        self.fc1   = nn.Linear(16*4*4, 100)  
        self.fc2   = nn.Linear(100, 7)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  
        x = torch.flatten(x, 1)                    
        x = F.relu(self.fc1(x))
        return self.fc2(x)                          


# Training now? 

def train_model(model, train_loader, test_loader, epochs=10): 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    
    for epoch in range(epochs): 
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader): 
            data, target = data.to(DEVICE), target.to(DEVICE)


if __name__ == "__main__": 
    set_seed(42)
    train_data, test_data = load_datasets()
    train_loader, test_loader = make_loaders(train_data, test_data)
    model = Tiny().to(DEVICE)
    train_model(model, train_loader, test_loader, epochs=10)