import torch, torchvision 
import torch.nn as nn 
import torch.nn.functional as F

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
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
    ])

def load_datasets(): 
    train_data = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform_data()) #or tfm_train/tfm_eval ? whats the difference here? 
    test_data = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform_data())
    
    labels_to_keep = torch.tensor([0, 1, 2, 3, 4, 5, 6])

    # Filter training data ... how can i choose the classes here? 
    train_mask = torch.isin(train_data.targets, labels_to_keep)
    train_data.data = train_data.data[train_mask]
    train_data.targets = train_data.targets[train_mask]

    # Filter test data
    test_mask = torch.isin(test_data.targets, labels_to_keep)
    test_data.data = test_data.data[test_mask]
    test_data.targets = test_data.targets[test_mask]
    
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs): 
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader): 
            data, target = data.to(DEVICE), target.to(DEVICE)


if __name__ == "__main__": 
    train_data, test_data = load_datasets()
    train_loader, test_loader = make_loaders(train_data, test_data)
    model = Tiny().to(DEVICE)
    train_model(model, train_loader, test_loader, epochs=10)