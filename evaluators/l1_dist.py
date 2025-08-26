import torch
import cv2
import numpy as np
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import Config

class L1Evaluator:

    def __init__(self, depth_model_path, body_model_path, hand_model_path, device="cuda"):
        depth_model_cfg = {"NAME": "MidasDetector", "PRETRAINED_MODEL": depth_model_path}
        depth_model_cfg = Config(load=False, cfg_dict=depth_model_cfg)
        self.depth_model = ANNOTATORS.build(depth_model_cfg).to(device)

        pose_model_cfg = {"NAME": "OpenposeAnnotator", "BODY_MODEL_PATH": body_model_path, "HAND_MODEL_PATH": hand_model_path}
        pose_model_cfg = Config(load=False, cfg_dict=pose_model_cfg)
        self.pose_model = ANNOTATORS.build(pose_model_cfg).to(device)

        self.device = device
    
    @torch.no_grad()
    def __call__(self, 
                 img=None, 
                 src_img=None, 
                 src_mask=None, 
                 img_type="raw",
                 **kwargs):
        res = dict()
        if img.size != src_img.size:
            img = img.resize(src_img.size)

        if img_type == "raw":
            img = np.array(img, dtype=np.float32)
            img = img / 255.
            src_img = np.array(src_img, dtype=np.float32)
            src_img = src_img / 255.
            if src_mask:
                src_mask = np.array(src_mask, dtype=np.uint8)
                src_mask = np.where(src_mask > 128, 1.0, 0.0)
                src_mask = np.stack([src_mask, src_mask, src_mask], axis=2)
                l1_score = (img - src_img) * (1 - src_mask)
                res['l1_raw'] = np.abs(l1_score).sum() / (1 - src_mask).sum()
            else:
                l1_score = img - src_img
                res['l1_raw'] = np.abs(l1_score).mean()
        elif img_type == "canny":
            canny_img = cv2.Canny(np.array(img, dtype=np.uint8), 100, 200)
            src_gray = np.asarray(src_img.convert("L"), np.uint8)
            canny_dist = np.abs(canny_img.astype(np.float32) - src_gray.astype(np.float32)).mean()
            res['l1_canny'] = 1.0 - canny_dist / 255.
        elif img_type == "depth":
            depth_img = self.depth_model(img)
            depth_gray = cv2.cvtColor(depth_img, cv2.COLOR_RGB2GRAY)
            src_gray = np.asarray(src_img.convert("L"), np.uint8)
            depth_dist = np.abs(depth_gray.astype(np.float32) - src_gray.astype(np.float32)).mean()
            res['l1_depth'] = 1.0 - depth_dist / 255.
        elif img_type == "pose":
            pose_img = self.pose_model(img)
            pose_gray = cv2.cvtColor(pose_img, cv2.COLOR_RGB2GRAY)
            src_gray = np.asarray(src_img.convert("L"), np.uint8)
            pose_dist = np.abs(pose_gray.astype(np.float32) - src_gray.astype(np.float32)).mean()
            res['l1_pose'] = 1.0 - pose_dist / 255.
        elif img_type == "colorize":
            img = img.convert("L")
            src_img = src_img.convert("L")
            gray_dist = np.abs(np.array(img, dtype=np.float32) - np.array(src_img, dtype=np.float32)).mean()
            res['l1_colorize'] = 1.0 - gray_dist / 255.
        else:
            raise ValueError(f"img_type {img_type} has not been supported yet.")
        return res