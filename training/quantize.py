import os

import numpy as np
import onnx
import torch
import torchvision
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


class MNISTCalibrationDataReader(CalibrationDataReader):

    def __init__(self, model_path, data_root="data", num_samples=1024, batch_size=32):
        super().__init__()
        self.model_path = model_path
        self.data_root = data_root
        self.num_samples = max(1, num_samples)
        self.batch_size = max(1, batch_size)

        self._input_name = None
        self._iter = None
        self._loader = None

        self._prepare()

    def _prepare(self):
        model = onnx.load(self.model_path)
        self._input_name = model.graph.input[0].name

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset = torchvision.datasets.MNIST(self.data_root, train=True, download=True, transform=transforms)

        indices = list(range(min(self.num_samples, len(dataset))))
        subset = torch.utils.data.Subset(dataset, indices)
        self._loader = torch.utils.data.DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )
        self._iter = None

    def _data_iterator(self):
        assert self._loader is not None and self._input_name is not None
        for images, _ in self._loader:
            for i in range(images.shape[0]):
                yield {self._input_name: images[i:i+1].numpy()}

    def get_next(self):
        if self._iter is None:
            self._iter = iter(self._data_iterator())
        try:
            return next(self._iter)
        except StopIteration:
            return None

    def rewind(self):
        self._iter = None


MODEL_INPUT_PATH = "artifacts/tiny_mnist_best.onnx"
MODEL_OUTPUT_PATH = "artifacts/tiny_mnist_best_quantized.onnx"
DATA_ROOT = "data"
NUM_SAMPLES = 1024
BATCH_SIZE = 32


def quantize():

    data_reader = MNISTCalibrationDataReader(
        model_path=MODEL_INPUT_PATH,
        data_root=DATA_ROOT,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
    )

    quantize_static(
        model_input=MODEL_INPUT_PATH,
        model_output=MODEL_OUTPUT_PATH,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
    )


if __name__ == "__main__":
    quantize()