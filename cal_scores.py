import json
import argparse
from collections import defaultdict



def aes(score_dict, **kwargs):
    logit = score_dict['aes_v2.5']
    return logit / 10.

def iqa(score_dict, **kwargs):
    logit = score_dict['musiq_koniq']
    return logit / 100.

def pf(score_dict, task, **kwargs):
    if task == "Image Colorize":
        clip_cap = score_dict['clip_cap']
        colorfulness = score_dict['colorfulness']
        return clip_cap * 2 * 0.75 + colorfulness * 0.25
    elif task == "Image Deblur":
        return iqa(score_dict)
    elif "vllmqa" in score_dict:
        clip_cap = score_dict['clip_cap']
        vllmqa = score_dict['vllmqa']
        return (clip_cap * 2 + vllmqa) / 2.
    else:
        clip_cap = score_dict['clip_cap']
        return clip_cap * 2
    
def sc(score_dict, **kwargs):
    clip_src = score_dict['clip_src']
    l1_raw = score_dict["l1_raw"]
    return (clip_src + 1 - l1_raw) / 2.

def rc(score_dict, task, **kwargs):
    if task in ["Face Reference Generation", "Face Swap"]:
        return score_dict['face_ref']
    elif task in ["Style Reference Generation", "Style Reference Editing"]:
        return score_dict['style_ref']
    elif task in ["Subject Reference Generation", "Subject-guided Inpainting", "Virtual Try On"]:
        return score_dict['dino_ref']
    else:
        print(task)
        return 0

def ctrl(score_dict, task, **kwargs):
    if task == "Image Deblur":
        ssim = score_dict['ssim']
        return ssim
    elif task == "Pose-guided Generation":
        l1_pose = score_dict['l1_pose']
        return l1_pose
    elif task == "Depth-guided Generation":
        l1_depth = score_dict['l1_depth']
        return l1_depth
    elif task == "Edge-guided Generation":
        l1_canny = score_dict['l1_canny']
        return l1_canny
    elif task == "Image Colorize":
        l1_colorize = score_dict['l1_colorize']
        return l1_colorize
    else:
        raise ValueError


def calculate_task_score(score_dict, task, **kwargs):
    if task=="Text2Image": return (aes(score_dict) + iqa(score_dict) + pf(score_dict, task)) / 3.
    if task=="Face Reference Generation": return (aes(score_dict) + iqa(score_dict) + pf(score_dict, task) + rc(score_dict, task)) / 4.
    if task=="Subject Reference Generation": return (aes(score_dict) + iqa(score_dict) + pf(score_dict, task) + rc(score_dict, task)) / 4.
    if task=="Style Reference Generation": return (aes(score_dict) + iqa(score_dict) + pf(score_dict, task) + rc(score_dict, task)) / 4.
    if task=="Color Editing": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Face Editing": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Style Editing": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Texture Editing": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Motion Editing": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Scene Editing": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Subject Addition": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Subject Change": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Subject Removal": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Text Removal": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Text Render": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Composite Editing": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Inpainting": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Outpainting": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Local Subject Addition": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Local Subject Removal": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Local Text Render": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Local Text Removal": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.3 * pf(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Virtual Try On": return 0.25 * aes(score_dict) + 0.25 * iqa(score_dict) + 0.2 * pf(score_dict, task) + 0.2 * rc(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Face Swap": return 0.25 * aes(score_dict) + 0.25 * iqa(score_dict) + 0.2 * pf(score_dict, task) + 0.2 * rc(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Subject-guided Inpainting": return 0.25 * aes(score_dict) + 0.25 * iqa(score_dict) + 0.2 * pf(score_dict, task) + 0.2 * rc(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Style Reference Editing": return 0.25 * aes(score_dict) + 0.25 * iqa(score_dict) + 0.2 * pf(score_dict, task) + 0.2 * rc(score_dict, task) + 0.1 * sc(score_dict)
    if task=="Pose-guided Generation": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.2 * pf(score_dict, task) + 0.2 * ctrl(score_dict, task)
    if task=="Depth-guided Generation": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.2 * pf(score_dict, task) + 0.2 * ctrl(score_dict, task)
    if task=="Edge-guided Generation": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.2 * pf(score_dict, task) + 0.2 * ctrl(score_dict, task)
    if task=="Image Deblur": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.2 * pf(score_dict, task) + 0.2 * ctrl(score_dict, task)
    if task=="Image Colorize": return 0.3 * aes(score_dict) + 0.3 * iqa(score_dict) + 0.2 * pf(score_dict, task) + 0.2 * ctrl(score_dict, task)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_file", "-f", type=str, default="results/ace/eval_result.txt")
    args = parser.parse_args()

    with open(args.score_file, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    task_sample_scores = dict()
    for line in lines:
        item_id, task, score_dict = line.split("#;#")
        score_dict = json.loads(score_dict)
        if task not in task_sample_scores:
            task_sample_scores[task] = defaultdict(list)
        for k, v in score_dict.items():
            task_sample_scores[task][k].append(v)
    
    task_scores = dict()
    for task, sample_scores in task_sample_scores.items():
        task_dim_scores = dict()
        for k, v in sample_scores.items():
            task_dim_scores[k] = sum(v) / len(v)

        task_scores[task] = calculate_task_score(task_dim_scores, task)
        print(f"Task: {task}, Task Score: {task_scores[task]}")
    
    collected_task_scores = list(task_scores.values())
    overall_score = sum(collected_task_scores) / len(collected_task_scores)
    print(f"Task num: {len(collected_task_scores)}, Overall Score: {overall_score}")