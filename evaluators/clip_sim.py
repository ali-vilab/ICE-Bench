import io
import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel


class CLIPEvaluator:

    def __init__(self, clip_path, device="cuda"):
        self.model = AutoModel.from_pretrained(clip_path, weights_only=False)
        self.processor = AutoProcessor.from_pretrained(clip_path)
        self.model = self.model.eval().requires_grad_(False).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, 
                 img=None,
                 src_img=None,
                 ref_img=None,
                 caption=None,
                 instruction=None,
                 clip_cap=False,
                 clip_src=False,
                 clip_ref=False,
                 **kwargs):

        assert (clip_cap or clip_src or clip_ref)
        instruction = instruction.replace("<SOURCE>", "source image").replace("<REF_1>", "reference image")

        res = dict()
        img_inputs = self.processor(images=img, return_tensors="pt")
        img_features = self.model.get_image_features(pixel_values=img_inputs['pixel_values'].to(self.device))
        img_features_norm = F.normalize(img_features, dim=-1)

        if clip_cap:
            assert caption is not None, "caption should not be None when calculating clip_cap score"
            cap_inputs = self.processor(text=[caption], padding="max_length", truncation=True, return_tensors="pt")
            cap_features = self.model.get_text_features(input_ids=cap_inputs['input_ids'].to(self.device))
            cap_features_norm = F.normalize(cap_features, dim=-1)
            clip_cap_sim = (img_features_norm @ cap_features_norm.T).item()
            res['clip_cap'] = clip_cap_sim
        
        if clip_src:
            assert src_img is not None, "src_img should not be None when calculating clip_src score"
            src_img_inputs = self.processor(images=src_img, return_tensors="pt")
            src_img_features = self.model.get_image_features(pixel_values=src_img_inputs['pixel_values'].to(self.device))
            src_img_features_norm = F.normalize(src_img_features, dim=-1)
            clip_src_sim = (img_features_norm @ src_img_features_norm.T).item()
            res['clip_src'] = clip_src_sim

        if clip_ref:
            assert ref_img is not None
            ref_img_inputs = self.processor(images=ref_img, return_tensors="pt")
            ref_img_features = self.model.get_image_features(pixel_values=ref_img_inputs['pixel_values'].to(self.device))
            ref_img_features_norm = F.normalize(ref_img_features, dim=-1)
            clip_ref_sim = (img_features_norm @ ref_img_features_norm.T).item()
            res['clip_ref'] = clip_ref_sim
        
        return res