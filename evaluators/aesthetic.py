import torch
import torch.nn as nn
from os import PathLike
from transformers import (
    SiglipImageProcessor,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers.image_processing_utils import BatchFeature
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention


def resize_img(img):
    w, h = img.size
    if h != 1024:
        new_h = 1024
        new_w = w * new_h / h
        new_img = img.resize((int(new_w), int(new_h)))
        return new_img
    else:
        return img


class AestheticPredictorV2_5Head(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.scoring_head = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.scoring_head(image_embeds)


class AestheticPredictorV2_5Model(SiglipVisionModel):
    PATCH_SIZE = 14

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.layers = AestheticPredictorV2_5Head(config)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple | ImageClassifierOutputWithNoAttention:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = super().forward(
            pixel_values=pixel_values,
            return_dict=return_dict,
        )
        image_embeds = outputs.pooler_output
        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        prediction = self.layers(image_embeds_norm)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct()

        if not return_dict:
            return (loss, prediction, image_embeds)

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=prediction,
            hidden_states=image_embeds,
        )


class AestheticPredictorV2_5Processor(SiglipImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> BatchFeature:
        return super().__call__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str
        | PathLike = "google/siglip-so400m-patch14-384",
        *args,
        **kwargs,
    ) -> "AestheticPredictorV2_5Processor":
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)




class AestheticEvaluator:

    def __init__(self, clip_path, mlp_path, device="cuda"):
        self.model = AestheticPredictorV2_5Model.from_pretrained(clip_path)
        self.processor = AestheticPredictorV2_5Processor.from_pretrained(clip_path)
        
        state_dict = torch.load(mlp_path, map_location=torch.device("cpu"), weights_only=False)
        self.model.layers.load_state_dict(state_dict, strict=True)
        self.model.eval().requires_grad_(False).to(torch.bfloat16).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, 
                 img=None, 
                 resize_to_1k=False,
                 **kwargs):
        if resize_to_1k:
            img = resize_img(img)
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(torch.bfloat16)
        pixel_values = pixel_values.to(self.device)
        aes_score = self.model(pixel_values).logits.squeeze().float().cpu().item()
        return {"aes_v2.5": aes_score}