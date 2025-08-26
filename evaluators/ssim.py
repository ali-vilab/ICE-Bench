import pyiqa
import torch
import torchvision.transforms.functional as TF


class SSIMEvaluator:
    def __init__(self, device="cuda"):
        self.model = pyiqa.create_metric('ssim', device=device)

    @torch.no_grad()
    def __call__(self, 
                img=None, 
                src_img=None,
                **kwargs):
        res = dict()
        if img.size != src_img.size:
            img = img.resize(src_img.size)

        img_tensor = TF.to_tensor(img)
        src_tensor = TF.to_tensor(src_img)

        ssim = self.model(img_tensor.unsqueeze(0), src_tensor.unsqueeze(0))
        res['ssim'] = ssim.cpu().item()
        return res