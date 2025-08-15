import torch, onnx, onnxruntime
from train import Tiny

def export(): 
    model = Tiny().to("cpu").eval()
    prev = torch.load("artifacts/tiny_mnist_best.pt", map_location="cpu")
    model.load_state_dict(prev["model_state_dict"])

    x = torch.randn(1, 1, 28, 28)
    torch.onnx.export(model, x, "artifacts/tiny_mnist_best.onnx", input_names=["input"], output_names=["logits"], opset_version=13)

    onnx.checker.check_model("artifacts/tiny_mnist_best.onnx")
    session = onnxruntime.InferenceSession("artifacts/tiny_mnist_best.onnx", providers=["CPUExecutionProvider"])
    y = session.run(None, {"input": x.numpy()})
    print("exported. Shape:", [o.shape for o in y])


if __name__ == "__main__":
    export()