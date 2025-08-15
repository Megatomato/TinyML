import torch, onnx, onnxruntime
from train import Tiny

def export(): 
    model = torch.load("artifacts/tiny_mnist_best.pt")
    model.eval()

    x = torch.randn(1, 1, 28, 28)
    torch.onnx.export(model, x, "artifacts/tiny_mnist_best.onnx", input_names=["input"], output_names=["logits"], opset_version=13)


if __name__ == "__main__":
    export()