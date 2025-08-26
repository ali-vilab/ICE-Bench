import os
import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import clip
import copy
from collections import OrderedDict
from torch.autograd import Function



def convert_weights_float(model: nn.Module):
    """Convert applicable model parameters to fp32"""

    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


## taken from https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/modules.py
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


def init_weights(m): # TODO: do we need init for layernorm?
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=1e-6)


class CSD_CLIP(nn.Module):
    """backbone + projection head"""
    def __init__(self, model_path, embedding_dim=1024, content_proj_head='default'):
        super(CSD_CLIP, self).__init__()

        clipmodel, _ = clip.load(model_path)
        self.backbone = clipmodel.visual
        self.embedding_dim = embedding_dim

        convert_weights_float(self.backbone)
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        self.last_layer_content = copy.deepcopy(self.backbone.proj)
        self.backbone.proj = None
        self.content_proj_head = content_proj_head


    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    def forward(self, input_data, alpha=None):
        feature = self.backbone(input_data)

        if alpha is not None:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
        else:
            reverse_feature = feature

        style_output = feature @ self.last_layer_style
        style_output = nn.functional.normalize(style_output, dim=1, p=2)

        content_output = reverse_feature @ self.last_layer_content
        content_output = nn.functional.normalize(content_output, dim=1, p=2)

        return feature, content_output, style_output
    



class StyleEvaluator:

    def __init__(self, clip_path, model_path, device="cuda"):
        self.model = CSD_CLIP(model_path=clip_path)

        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        new_state_dict = OrderedDict()
        for k, v in state_dict['model_state_dict'].items():
            if k.startswith("module."):
                k = k.replace("module.", "")
            new_state_dict[k] = v

        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"=> loaded checkpoint with msg {msg}")

        self.model = self.model.eval().requires_grad_(False).to(device)

        self.transforms = T.Compose([
                T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        
        self.device = device

    @torch.no_grad()
    def __call__(self, 
                img=None, 
                ref_img=None,
                **kwargs):
        img = self.transforms(img)
        img = img.unsqueeze(0).to(self.device)
        _, _, img_features = self.model(img)

        ref_img = self.transforms(ref_img)
        ref_img = ref_img.unsqueeze(0).to(self.device)
        _, _, ref_img_features = self.model(ref_img)
        score_ref = (img_features @ ref_img_features.T).cpu().item()
        
        return {"style_ref": score_ref}