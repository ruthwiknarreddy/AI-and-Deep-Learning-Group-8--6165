import torch
import os

## Pytorch has pretrained AlexNet while Keras does not ##
## I need to import the torch model ##
if not os.path.isdir("alexnet_tf"):
    import onnx
    from onnx_tf.backend import prepare
    import torchvision.models as models

    model = models.alexnet(pretrained = True)
    model.eval()
    random_input = torch.rand(1,3,224,224)
    torch.onnx.export(model, random_input, "alexnet.onnx", input_names=["input"],
		output_names=["output"], opset_version=11)
    onnx_alexnet = onnx.load("alexnet.onnx")
    tf_alexnet = prepare(onnx_alexnet)
    tf_alexnet.export_graph("alexnet_tf") ## usable by keras