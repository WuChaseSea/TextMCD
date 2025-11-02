# TextMCD

Official implementation for the paper "TextMCD: Mask Classification Based Change Detection Network Combining Multimodal Vision-Language Supervised Learning"

## Usage

### Install

* Create a conda virtual environment and activate it:

```sh
conda create -n ecd python=3.10 -y
conda activate ecd
pip install -r requirements.txt
```

### Data Preparation

Dataset download link:

* LEVIR-CD [https://justchenhao.github.io/LEVIR/](https://justchenhao.github.io/LEVIR/)
* WHUCD [https://gpcv.whu.edu.cn/data/building_dataset.html](https://gpcv.whu.edu.cn/data/building_dataset.html)
* CLCD [https://github.com/liumency/CropLand-CD](https://github.com/liumency/CropLand-CD)
* SECOND [https://captain-whu.github.io/SCD/](https://captain-whu.github.io/SCD/)

After downloading and processing, move it to ./data folder. This folder path can be modified in corresponding config file.

### Model Preparation

Pretrained model file download link:

CLIP [https://github.com/open-mmlab/mmpretrain/tree/main/configs/clip](https://github.com/open-mmlab/mmpretrain/tree/main/configs/clip)

Move pretrained model file to ./pretraned_models, this folder path can also be modified in corresponding config file.

## Inference

```sh
python application/cd_application.py --config application/predict_pred.yaml
```

## Train

```sh
python Solve.py --config configs/textmcd/cd_mmseg_clcd.yaml --gpus 0
```

## TODO

* organize the model files of textmcd;
* update pretrained models;

✅ 2025/11/02 upload some training files for the framework.
