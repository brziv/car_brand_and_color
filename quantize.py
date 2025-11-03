from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = "enetv2s.onnx"
output_model = "enetv2s_int8.onnx"

quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    per_channel=True,
    weight_type=QuantType.QInt8
)

print("Quantized.")
