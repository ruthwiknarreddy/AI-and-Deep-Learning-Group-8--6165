import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    Selvaraju et al. (2017) - https://arxiv.org/abs/1610.02391

    Hooks into a target convolutional layer, captures activations on the forward pass
    and gradients on the backward pass, then produces a class-discriminative heatmap.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        # Register forward hook — also attaches a tensor-level gradient hook
        # on the activation output to avoid conflicts with inplace ReLU ops.
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()
        # Attach gradient hook directly to the output tensor (not the module).
        # This works correctly even when downstream ops use inplace=True.
        output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad.detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: preprocessed image tensor [1, C, H, W]
            class_idx: class index to explain (None = argmax / binary output)

        Returns:
            cam: numpy array [H', W'] normalized to [0, 1]
            class_idx: predicted or requested class index (None for binary)
        """
        self.model.eval()

        output = self.model(input_tensor)

        # ---- determine the score to backprop ----
        is_binary = (output.dim() == 1) or (output.shape[-1] == 1)
        if is_binary:
            score = output.squeeze()
            class_idx = None
        else:
            if class_idx is None:
                class_idx = int(output.argmax(dim=1).item())
            score = output[0, class_idx]

        self.model.zero_grad()
        score.backward()

        # ---- Grad-CAM formula ----
        # weights = global average pool of gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)

        # normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy(), class_idx

    def overlay_on_image(self, cam, original_image_np, alpha=0.45):
        """
        Resize the CAM to match the original image and overlay as a color heatmap.

        Args:
            cam: numpy array [H', W'] in [0, 1]
            original_image_np: numpy array [H, W, 3] in [0, 255] uint8 or [0, 1] float
            alpha: heatmap opacity

        Returns:
            overlay: blended image [H, W, 3] uint8
            heatmap: pure heatmap [H, W, 3] uint8
        """
        h, w = original_image_np.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        if original_image_np.max() <= 1.0:
            img_uint8 = (original_image_np * 255).astype(np.uint8)
        else:
            img_uint8 = original_image_np.astype(np.uint8)

        overlay = (alpha * heatmap_rgb + (1 - alpha) * img_uint8).astype(np.uint8)
        return overlay, heatmap_rgb

    def remove_hooks(self):
        self._fwd_hook.remove()


def get_target_layer(model, arch):
    """
    Returns the last convolutional layer for Grad-CAM visualization.

    AlexNet  → model.features[10]  (last Conv2d, 256 channels)
    GoogLeNet → model.inception5b  (last Inception module, 1024 channels)
    """
    arch = arch.lower()
    if arch == "alexnet":
        return model.features[10]
    elif arch == "googlenet":
        return model.inception5b
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose 'alexnet' or 'googlenet'.")
