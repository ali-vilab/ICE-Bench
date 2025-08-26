from functools import partial
from evaluators import *

task2metric = {
    "Text2Image": ["aesthetic", "imaging", "clip_cap"],
    "Face Reference Generation": ["aesthetic", "imaging", "clip_cap", "clip_ref", "face_ref"],
    "Subject Reference Generation": ["aesthetic", "imaging", "clip_cap", "clip_ref", "dino_ref"],
    "Style Reference Generation": ["aesthetic", "imaging", "clip_cap", "clip_ref", "style_ref"],
    "Color Editing": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Face Editing": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Style Editing": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Texture Editing": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Motion Editing": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Scene Editing": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Subject Addition": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Subject Change": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Subject Removal": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Text Removal": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Text Render": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Composite Editing": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Inpainting": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Outpainting": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Local Subject Addition": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Local Subject Removal": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Local Text Render": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Local Text Removal": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa"],
    "Virtual Try On": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa", "dino_ref"],
    "Face Swap": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa", "face_ref"],
    "Subject-guided Inpainting": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa", "dino_ref"],
    "Style Reference Editing": ["aesthetic", "imaging", "clip_cap", "clip_src", "l1_raw", "vllmqa", "style_ref"],
    "Pose-guided Generation": ["aesthetic", "imaging", "clip_cap", "l1_pose"],
    "Depth-guided Generation": ["aesthetic", "imaging", "clip_cap", "l1_depth"],
    "Edge-guided Generation": ["aesthetic", "imaging", "clip_cap", "l1_canny"],
    "Image Deblur": ["aesthetic", "imaging", "ssim"],
    "Image Colorize": ["aesthetic", "imaging", "clip_cap", "l1_colorize", "colorfulness"]
}

model_dir = "models"

evaluators = dict()
evaluators['AES'] = AestheticEvaluator(clip_path=f"{model_dir}/siglip-so400m-patch14-384/", mlp_path=f"{model_dir}/aesthetic_predictor_v2_5.pth")
evaluators['IMG'] = ImagingEvaluator(model_path=f"{model_dir}/musiq_koniq_ckpt-e95806b9.pth")
evaluators['CLIP'] = CLIPEvaluator(clip_path=f"{model_dir}/clip-vit-large-patch14-336/")
evaluators['DINO'] = DINOEvaluator(model_path=f"{model_dir}/dinov2-giant/")
evaluators['FACE'] = FaceEvaluator(model_dir="./")
evaluators['STYLE'] = StyleEvaluator(clip_path=f"{model_dir}/ViT-L-14.pt", model_path=f"{model_dir}/csd_vit_l.pth")
evaluators['L1'] = L1Evaluator(depth_model_path=f"{model_dir}/dpt_hybrid-midas-501f0c75.pt", body_model_path=f"{model_dir}/body_pose_model.pth", hand_model_path=f"{model_dir}/hand_pose_model.pth")
evaluators['SSIM'] = SSIMEvaluator()
evaluators['COLOR'] = ColorfulEvaluator()
evaluators['VLLM'] = VllmQA(model_path=f"{model_dir}/Qwen2.5-VL-72B-Instruct/")


metric2evaluator = dict()
metric2evaluator['aesthetic'] = partial(evaluators['AES'].__call__, resize_to_1k=True)
metric2evaluator['imaging'] = partial(evaluators['IMG'].__call__, resize_to_1k=True)
metric2evaluator['clip_cap'] = partial(evaluators['CLIP'].__call__, clip_cap=True)
metric2evaluator['clip_src'] = partial(evaluators['CLIP'].__call__, clip_src=True)
metric2evaluator['clip_ref'] = partial(evaluators['CLIP'].__call__, clip_ref=True)
metric2evaluator['dino_ref'] = evaluators['DINO'].__call__
metric2evaluator['style_ref'] = evaluators['STYLE'].__call__
metric2evaluator['face_ref'] = evaluators['FACE'].__call__
metric2evaluator['l1_raw'] = partial(evaluators['L1'].__call__, img_type='raw')
metric2evaluator['l1_pose'] = partial(evaluators['L1'].__call__, img_type='pose')
metric2evaluator['l1_depth'] = partial(evaluators['L1'].__call__, img_type='depth')
metric2evaluator['l1_canny'] = partial(evaluators['L1'].__call__, img_type='canny')
metric2evaluator['l1_colorize'] = partial(evaluators['L1'].__call__, img_type='colorize')
metric2evaluator['ssim'] = evaluators['SSIM'].__call__
metric2evaluator['colorfulness'] = evaluators['COLOR'].__call__
metric2evaluator['vllmqa'] = evaluators['VLLM'].__call__