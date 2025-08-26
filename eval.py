import json
import argparse
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from config import task2metric, metric2evaluator


parser = argparse.ArgumentParser()
parser.add_argument("--meta_file", "-m", type=str, required=True, default="data/data.jsonl")
parser.add_argument("--result_file", "-f", type=str, required=True)
parser.add_argument("--save_file", "-s", type=str, required=True)
args = parser.parse_args()


item_metas = defaultdict(dict)
with open(args.meta_file, 'r') as f:
    task_samples = f.read().strip().split("\n")
    for i in task_samples:
        item = json.loads(i)
        item_id = item['ItemID']
        item_metas[item_id] = item

with open(args.result_file, 'r') as f:
    item_results = json.load(f)


item_scores = dict()
for item_id, item_meta in tqdm(item_metas.items()):
    task = item_meta['Task']
    metrics = task2metric[task]
    src_img_path = item_meta['SourceImage']
    src_mask_path = item_meta['SourceMask']
    ref_img_paths = item_meta['ReferenceImages']
    instruction = item_meta['Instruction']
    caption = item_meta['TargetCaption']
    tgt_img_path = item_results[item_id]

    src_img = Image.open(src_img_path).convert('RGB') if len(src_img_path) > 0 else None
    src_mask = Image.open(src_mask_path).convert('L') if len(src_mask_path) > 0 else None
    ref_img = Image.open(ref_img_paths[0]).convert('RGB') if len(ref_img_paths) > 0 else None
    tgt_img = Image.open(tgt_img_path).convert('RGB')

    inputs = {
        "img": tgt_img,
        "src_img": src_img,
        "src_mask": src_mask,
        "ref_img": ref_img,
        "caption": caption,
        "instruction": instruction
    }
    
    item_score = dict()
    for metric in metrics:
        if metric not in metric2evaluator:
            continue
        item_score.update(metric2evaluator[metric](**inputs))

    item_scores[item_id] = (task, item_score)


with open(args.save_file, 'w') as f:
    for item_id, (task, item_score) in item_scores.items():
        for key, value in item_score.items():
            item_score[key] = float(value)
        f.writelines(item_id + "#;#" + task + "#;#" + json.dumps(item_score, ensure_ascii=False) + "\n")
