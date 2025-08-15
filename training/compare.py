import os
import json
import time

import numpy as np
import torch
import torchvision
import onnxruntime as ort


def transform_eval():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])


def load_test_dataset(data_root = "data", download = True):
    classes_to_keep = list(range(7)) 
    tfm_eval = transform_eval()
    test_full = torchvision.datasets.MNIST(data_root, train=False, download=download, transform=tfm_eval)

    keep = torch.tensor(classes_to_keep)
    test_idx = torch.nonzero(torch.isin(test_full.targets, keep)).squeeze(1).tolist()
    test_subset = torch.utils.data.Subset(test_full, test_idx)
    return test_subset


def make_loader(dataset, batch_size = 64, workers = 2):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def evaluate_onnx_model(model_path, loader, provider = "CPUExecutionProvider"):
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    size_bytes = os.path.getsize(model_path)

    try:
        session = ort.InferenceSession(model_path, providers=[provider])
        input_name = session.get_inputs()[0].name
    except Exception as e:
        return {
            "path": model_path,
            "provider": provider,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 4),
            "accuracy": None,
            "avg_latency_ms_per_image": None,
            "total_inference_images": 0,
            "inference_supported": False,
            "error": str(e),
        }

    total_correct = 0
    total_samples = 0
    total_inference_s = 0.0

    
    for images, labels in loader:
        labels_np = labels.numpy()
        batch_size = images.shape[0]

        batch_preds = np.empty((batch_size,), dtype=np.int64)

        for idx in range(batch_size):
            image_np = images[idx:idx + 1].numpy() 
            start = time.perf_counter()
            outputs = session.run(None, {input_name: image_np})
            total_inference_s += (time.perf_counter() - start)
            logits = outputs[0]  
            batch_preds[idx] = int(np.argmax(logits, axis=1)[0])

        total_correct += int((batch_preds == labels_np).sum())
        total_samples += batch_size

    accuracy = total_correct / max(total_samples, 1)
    avg_latency_ms_per_image = (total_inference_s / max(total_samples, 1)) * 1000.0

    return {
        "path": model_path,
        "provider": provider,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 4),
        "accuracy": round(accuracy, 6),
        "avg_latency_ms_per_image": round(avg_latency_ms_per_image, 6),
        "total_inference_images": total_samples,
        "inference_supported": True,
    }


def compare_and_save():
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    fp32_model = os.path.join(artifacts_dir, "tiny_mnist_best.onnx")
    int8_model = os.path.join(artifacts_dir, "tiny_mnist_best_quantized.onnx")

    test_ds = load_test_dataset(data_root="data", download=True)
    test_loader = make_loader(test_ds, batch_size=128, workers=2)

    fp32_metrics = evaluate_onnx_model(fp32_model, test_loader)
    int8_metrics = evaluate_onnx_model(int8_model, test_loader)

    size_reduction_pct = 0.0
    if fp32_metrics["size_bytes"] > 0:
        size_reduction_pct = 100.0 * (fp32_metrics["size_bytes"] - int8_metrics["size_bytes"]) / fp32_metrics["size_bytes"]

    latency_speedup_x = None
    if (
        isinstance(fp32_metrics.get("avg_latency_ms_per_image"), (int, float)) and
        isinstance(int8_metrics.get("avg_latency_ms_per_image"), (int, float)) and
        int8_metrics["avg_latency_ms_per_image"] > 0
    ):
        latency_speedup_x = fp32_metrics["avg_latency_ms_per_image"] / int8_metrics["avg_latency_ms_per_image"]

    comparison = {
        "non_quantized": fp32_metrics,
        "quantized": int8_metrics,
        "delta": {
            "size_reduction_percent": round(size_reduction_pct, 4),
            "accuracy_diff": (
                round(int8_metrics["accuracy"] - fp32_metrics["accuracy"], 6)
                if isinstance(fp32_metrics.get("accuracy"), (int, float)) and isinstance(int8_metrics.get("accuracy"), (int, float))
                else None
            ),
            "latency_speedup_x": round(latency_speedup_x, 6) if isinstance(latency_speedup_x, (int, float)) else None,
        },
    }

    out_path = os.path.join(artifacts_dir, "model_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"Wrote comparison to {out_path}")


if __name__ == "__main__":
    compare_and_save()
