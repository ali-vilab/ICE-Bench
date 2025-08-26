import os
import io
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from insightface.app import FaceAnalysis

class FaceEvaluator:
    def __init__(self, model_dir, device="cuda"):        
        self.models = []
        self.det_sizes = [640, 480, 384, 256, 128]
        for det_size in self.det_sizes:
            model = FaceAnalysis(name="buffalo_l", root=model_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            model.prepare(ctx_id=0, det_size=(det_size, det_size))
            self.models.append(model)

    @torch.no_grad()
    def __call__(self, 
                 img=None, 
                 ref_img=None,
                 **kwargs):
        img = np.array(img)
        img_faces = []
        for model in self.models:
            img_faces = model.get(img)
            if len(img_faces) > 0:
                break
        
        if len(img_faces) == 0:
            return {"face_ref": 0}
        
        img_features = np.stack([i.normed_embedding for i in img_faces], axis=0)
        
        ref_img = np.array(ref_img)
        ref_faces = []
        for model in self.models:
            ref_faces = model.get(ref_img)
            if len(ref_faces) > 0:
                break
        
        if len(ref_faces) == 0:
            return {"face_ref": 0}
        
        ref_img_features = np.expand_dims(ref_faces[0].normed_embedding, axis=0)
        face_ref = (ref_img_features @ img_features.T).max()
        return {"face_ref": face_ref}