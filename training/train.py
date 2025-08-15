import torch, torchvision 
import torch.nn as nn 
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import random_split, Subset

def set_device(): 
    if torch.cuda.is_available(): 
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        return torch.device("mps")
    else: 
        return torch.device("cpu") 
    
DEVICE = set_device() 

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



def load_datasets(data_root: str = "data", download: bool = True, val_ratio: float = 0.10, seed: int = 1337):
    classes = list(range(7))  # keep digits 0..6
    tfm_train = transform_train()
    tfm_eval  = transform_eval()

    train_full = torchvision.datasets.MNIST(data_root, train=True,  download=download, transform=tfm_train)
    test_full  = torchvision.datasets.MNIST(data_root, train=False, download=download, transform=tfm_eval)

    keep = torch.tensor(classes)
    train_idx = torch.nonzero(torch.isin(train_full.targets, keep)).squeeze(1).tolist()
    test_idx  = torch.nonzero(torch.isin(test_full.targets,  keep)).squeeze(1).tolist()

    train_subset = Subset(train_full, train_idx)
    test_subset  = Subset(test_full,  test_idx)

    g = torch.Generator().manual_seed(seed)
    val_len   = max(1, int(len(train_subset) * val_ratio))
    train_len = len(train_subset) - val_len
    train_ds, val_ds = random_split(train_subset, [train_len, val_len], generator=g)

    return train_ds, val_ds, test_subset

def make_loaders(train_ds, val_ds, test_ds, batch_size=64, workers=2):

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )

    return train_loader, val_loader, test_loader

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


def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                    device: torch.device, clip_grad: float | None = None):

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device).long()

        optimizer.zero_grad()
        logits = model(x)

        loss = criterion(logits, y)
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        pred = logits.argmax(dim=1)
        running_correct += (pred == y).sum().item()
        running_total += y.size(0)
        running_loss += loss.item() * y.size(0)

    avg_loss = running_loss / max(running_total, 1)
    avg_acc  = running_correct / max(running_total, 1)
    return avg_loss, avg_acc

def evaluate( model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device): 
    model.eval()
    
    with torch.no_grad():
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long()

            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_total += y.size(0)


    avg_loss = running_loss / max(running_total, 1)
    avg_acc  = running_correct / max(running_total, 1)
    return avg_loss, avg_acc

if __name__ == "__main__": 
    set_seed(42)

    train_ds, val_ds, test_ds = load_datasets()
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size=64, workers=2)

    model = Tiny().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    best_val_acc = 0.0
    best_state = None

    print(f"Using device: {DEVICE}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, clip_grad=1.0)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"]) 
        torch.save(best_state, "artifacts/tiny_mnist_best.pt")

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"loss={test_loss:.4f} acc={test_acc:.4f}")