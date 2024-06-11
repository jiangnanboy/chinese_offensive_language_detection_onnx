Chinese Offensive Language Detection using onnx model

利用onnx格式进行中文冒犯语言检测

- label 0: safe,
- label 1: offensive

- model2onnx -> src/model2onnx.py/convert2onnx
- onnxtest -> src/model2onnx.py/onnx_test

- https://huggingface.co/thu-coai/roberta-base-cold
- https://github.com/thu-coai/COLDataset
