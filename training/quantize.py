import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize(): 
    input_model_path = "artifacts/tiny_mnist_best.onnx"
    preprocessed_model_path = "artifacts/tiny_mnist_best_preprocessed.onnx"
    quantized_model_path = "artifacts/tiny_mnist_best_quantized.onnx"

    model_for_quantization = input_model_path

    quantize_dynamic(
        model_for_quantization,
        quantized_model_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["Gemm", "MatMul"],
    )


if __name__ == "__main__":
    quantize()