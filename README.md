# Spatio-Temporal MultiObject Tracking Transformer

The repository is an official implementation of the paper [End-to-End Multiple Object Detection and Tracking with Space-Time Transformers]().

## Introduction

STMOTR based on the saptio-temporal features of video recognition instances is a novel end-to-end framework for multiple-object tracking.

![Construction](https://s2.loli.net/2023/06/28/jO25ckqTzgiHedG.png)

**Abstract**. The Multiple Object Tracking (MOT) is a fundamental task in intelligent transportation. Its key challenge is spatio-temporal features fusion of object along track. Existing tracking-by-detection(TBD) methods adopt temporal modeling of detection results to achieve the association of track query over frames. However, these methods ignore the direct association of track query on spatio-temporal features of video understanding. In the paper, we propose a novel end-to-end multiple-object tracking network with spatio-temporal Transformers (STMOTR). The proposed network utilizes Swin Transformer as the backbone to extract spatio-temporal features of track query. And process these features with a reconstructed Deformerable-DETR network. We evaluate the proposed method on UA-DETRAC datasets with $39.8\%$ PR-MOTA and Tunnel datasets with $79.6\%$ MOTA. The experimental results show that the proposed method can achieve state-of-the-art performance with real-time speed.

## Main Results

### UA-DETRAC

| Method | PR-MOTA | PR-MOTP | PR-IDS | PR-MT | PR-ML | FPS | URL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
|STMOTR | 39.8% | 88.6% | 36.1 | 30.3% | 30.4% | 11.96 | [model](https://drive.google.com/file/d/1_BX8nsaQV4WtCI4o9FlCUSWFMhL_eF0D/view?usp=drive_link) |

![result show](https://s2.loli.net/2023/06/28/gy64ekBWRbxPl98.gif)

### Tunnel

| Method | MOTA | MOTP | IDS | MT | ML | IDF1 | URL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| STMOTR | 79.6% | 0.169 | 227 | 475 | 47 | 82.4% | [model](https://drive.google.com/file/d/1WEko9ygW2Yg7VmXvLgPNUKaKqCNOvkPu/view?usp=drive_link) |

*Note:*

1. Trained on 4 NVIDIA Tesla V100 GPUs;
2. The training time for UA-DETRAC is about 7 days on V100 with batch size 4;
3. The inference speed is about 11.96 FPS for resolution 960x540 with batch size 1.
4. All model of STMOTR are trained with Video Swin tiny Transformer as backbone.

## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Requirements

- pytorch >= 1.12.1, torchviisionn>=0.13.1
  
  We recommend you to use Anaconda to create a conda environment:
  ```bash
  conda create -n stmotr python=3.9
  conda activate stmotr
  conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
  ```

- Other requirements
  
  ```bash
  conda install -c conda-forge opencv matplotlib scipy pandas psutil tqdm simplejson einops pyyaml motmetrics tabulate
  pip install tensorboard moviepy
  ```

- If the usage of deformableDETR, you should install `CUDA Functions of Multi-Scale Deformable Attention`

  ```shell
  pip install MultiScaleDeformableAttention
  ```

## Usage

### Data Preparation

1. Download UA-DETRAC dataset from [UA-DETRAC](https://detrac-db.rit.albany.edu/download) and custom MOT-like dataset like TUNNEL and organize them as following:

```
.
├── UA_DETRAC
│   ├── test
│       |── DETRAC-Annotations-XML
|       |── Insight-MVT_Annotation
│   └── train
│       |── DETRAC-Annotations-XML
|       |── Insight-MVT_Annotation
├── TUNNEL
│   ├── test
│       |── $VideoName
|           |── gt
|           |── img1
|           |── seqinfo.ini
│   └── train
│       |── $VideoName
|           |── gt
|           |── img1
|           |── seqinfo.ini

```

*Note:*
The trained or validated video name should be written in `sequence_list_train.txt` or `sequence_list_test.txt` in the corresponding dataset folder.

### Training and Evaluation

All training & evaluation configurations are in `configs` folder. You can modify the configuration file to train or evaluate the model.

#### Training on single node with mulitple gpus

To train on a single machine with 3 gpus for 50 epochs run:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2
python main.py --running_mode train --shard_id 0 --num_shards 1 --num_gpus 3 --cfg configs/ua_destmotr_train.yaml
```

#### Evaluation on UA-DETRAC

```shell
python main.py --running_mode eval --cfg configs/ua_destmotr_eval.yaml
```

#### Ablation on UA-DETRAC

```shell
python main.py --running_mode ablation --cfg configs/ua_destmotr_abla.yaml
```

#### Test on Custom Video

You should reconstrution the video as imgs file save and build custom dataset like TUNNEL.
```shell
python main.py --running_mode eval --cfg configs/ua_destmotr_pred.yaml
```

## Citing STMOTR

If you find STMOTR useful in your research, please consider citing:

```bibtex
```
