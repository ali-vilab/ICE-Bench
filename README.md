# ICE-Bench: A Unified and Comprehensive Benchmark for Image Creating and Editing

<font size=3><div align='center' > [[🍎 Project Page](https://ali-vilab.github.io/ICE-Bench-Page/)] [[📖 arXiv Paper](https://arxiv.org/abs/2503.14482)] [[🤗 Dataset](https://huggingface.co/datasets/ali-vilab/ICE-Bench)] </div></font>

---

## 🔥 News

* **`2025.8.26`** The code and dataset for automated evaluation is available now.
* **`2025.6.26`** Our paper has been received by ICCV 2025!
* **`2025.3.18`** Paper is available on Arxiv. 


## Abstract

<p align="center">
    <img src="./assets/teaser.png" height="100%">
</p>




## Evaluation


### 1. Environment Setup

Set up the environment for running the evaluation scripts.

```bash
conda env create -f environment.yml
```

### 2. Download and Prepare the Dataset and Models

Download the evaluation data and models from [Hugging Face repo](https://huggingface.co/datasets/ali-vilab/ICE-Bench).
Then unzip `data.zip` and`models.zip` under the root of ICE-Bench project.

For Qwen2.5-VL-72B-Instruct, you should download it from the [official repo](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) and place it in the `models` folder under the root of this project.

### 3. Run your Model to Generate Results

Run your model to generate the results for all tasks. Save the generated images in the `results/{METHOD_NAME}/images` folder, 
and keep an json file that contains (item_id, image_save_path) key-value pairs.

Your directory structure should look like this:

```
ICE-Bench/
├── assets/
├── dataset/
│    ├── images/
│    └── data.jsonl
├── models/
│    ├── Qwen2.5-VL-72B-Instruct
│    ├── aesthetic_predictor_v2_5.pth
│    └── ...
├── results/
│    └── method_name/
│       ├── images/
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       └── gen_info.json
├── evaluators/
├── config.py
├── requirements.txt
├── cal_scores.py
├── eval.py
└── ...
```

The `gen_info.json` file look like this:

```
{
    "item_id1": "results/{METHOD}/images/image1.jpg",
    "item_id2": "results/{METHOD}/images/image2.jpg",
    ...
}
```


### 4. Run Evaluation

```bash
python eval.py -m dataset/data.jsonl -f results/{METHOD}/gen_info.json -s results/{METHOD}/eval_result.txt
```

The evaluation results will be saved in the `results/{METHOD}/eval_result.txt` file.

### 5. Calculate Task Scores and Method Scores

```bash
python cal_scores.py -f results/{METHOD}/eval_result.txt
```


## Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
@article{pan2025ice,
  title={Ice-bench: A unified and comprehensive benchmark for image creating and editing},
  author={Pan, Yulin and He, Xiangteng and Mao, Chaojie and Han, Zhen and Jiang, Zeyinzi and Zhang, Jingfeng and Liu, Yu},
  journal={arXiv preprint arXiv:2503.14482},
  year={2025}
}
```