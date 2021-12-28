import onnx

model = '../data/inceptionV3.onnx'
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model)), model)
