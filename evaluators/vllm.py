import os
import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class VllmQA:
    def __init__(self, model_path, device="cuda"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device

    @torch.no_grad()
    def __call__(self, 
                 img=None, 
                 src_img=None, 
                 src_mask=None, 
                 ref_img=None, 
                 instruction=None, 
                 **kwargs):
        min_pixels = 512*512
        max_pixels = 1024*1024
        torch.cuda.empty_cache()
        if src_mask is not None and ref_img is not None:
            text = f"现在有四张图，其中第一张图为[SOURCE]图，第二张图是对第一张图进行图片编辑得到的，第三张图是规定编辑区域的掩码图，只能在掩码图指定的范围内进行图像编辑，第四张图是[REF]图。编辑指令是：'{instruction}'。在掩码图指定的范围内，需要按照指令的要求，参考[REF]图的内容，对[SOURCE]图进行编辑，其他部分不允许发生改变。请判断从第一张图到第二张图的变化是否与编辑指令和掩码图一致。如果变化与指令一致请输出1，不一致请输出0。如果掩码图指定区域以外的部分发生改变了也输出0。请直接输出结果，不需要给出解释"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": src_img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "image", "image": src_mask, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "image", "image": ref_img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "text", "text": text},
                    ],
                }
            ]
        elif ref_img is not None:
            text = f"现在有三张图，其中第一张图为[SOURCE]图，第二张图是对第一张图进行图片编辑得到的，第三张图是[REF]图。编辑指令是：'{instruction}'。需要按照指令的要求，参考[REF]图的内容，对[SOURCE]图进行编辑。请判断从第一张图到第二张图的变化是否与编辑指令一致。如果变化与指令一致请输出1，不一致请输出0。请直接输出结果，不需要给出解释"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": src_img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "image", "image": ref_img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "text", "text": text},
                    ],
                }
            ]
        elif src_mask is not None:
            text = f"现在有三张图，其中第一张图为[SOURCE]图，第二张图是对第一张图进行图片编辑得到的，第三张图是规定编辑区域的掩码图，只能在掩码图指定的范围内进行图像编辑。编辑指令是：'{instruction}'。在掩码图指定的范围内，除了指令提到的变化，其他部分不允许发生改变。请判断从第一张图到第二张图的变化是否与编辑指令和掩码图一致。如果变化与指令一致请输出1，不一致请输出0。如果掩码图指定区域以外的部分发生改变了也输出0。请直接输出结果，不需要给出解释"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": src_img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "image", "image": src_mask, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "text", "text": text},
                    ],
                }
            ]
        else:
            text = f"现在有两张图，其中第一张图为[SOURCE]图，第二张图是将第一张图进行图片编辑得到的，编辑指令是：'{instruction}'。请判断从左图到右图的变化是否与编辑指令一致，除了指令提到的变化，其他部分不允许发生改变。如果变化与指令一致请输出1，不一致请输出0。其他部分发生改变了也输出0。请直接输出结果，不需要给出解释"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": src_img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "text", "text": text},
                    ],
                }
            ]
        input_text = self.processor.apply_chat_template(messages, 
                                                        tokenize=False, 
                                                        add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[input_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids_trimmed, 
                                                skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=False)
        response = response[0]
        return {"vllmqa": float(response.replace("\n", "").replace(" ", ""))}