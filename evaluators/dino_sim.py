import io
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel



class DINOEvaluator:
    def __init__(self, model_path, device="cuda"):
        self.model = AutoModel.from_pretrained(model_path, weights_only=False)
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = self.model.eval().requires_grad_(False).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, 
                 img=None, 
                 src_mask=None,
                 ref_img=None, 
                 **kwargs):
        if src_mask is not None:
            if src_mask.size != img.size:
                src_mask = src_mask.resize(img.size)
            src_mask = np.array(src_mask, dtype=np.float32)
            src_mask = np.where(src_mask > 128, 1.0, 0.0)
            src_mask = np.stack([src_mask, src_mask, src_mask], axis=2)

            _x = np.asarray(img, dtype=np.float32)
            _x = _x * src_mask + np.ones_like(_x) * 255. * ( 1 - src_mask)
            img = _x.astype(np.uint8)
            img = Image.fromarray(img)
        
        img_inputs = self.processor(images=img, return_tensors="pt")
        outputs = self.model(pixel_values=img_inputs['pixel_values'].to(self.device))
        img_features = outputs.last_hidden_state.mean(dim=1)
        img_features = F.normalize(img_features, dim=-1)
        
        ref_img_inputs = self.processor(images=ref_img, return_tensors="pt")
        outputs = self.model(pixel_values=ref_img_inputs['pixel_values'].to(self.device))
        ref_img_features = outputs.last_hidden_state.mean(dim=1)
        ref_img_features = F.normalize(ref_img_features, dim=-1)
        dino_ref = (img_features @ ref_img_features.T).item()
        
        return {"dino_ref": dino_ref}