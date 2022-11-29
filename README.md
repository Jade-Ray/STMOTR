# Multimodal MultiObject Tracking Transformer

## Require

```shell
conda create -n mmotr python=3.9
conda activate mmotr

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

conda install -c conda-forge opencv matplotlib scipy pandas psutil tqdm simplejson einops pyyaml motmetrics tabulate

pip install tensorboard moviepy
```

- If the usage of deformableDETR, you should install `CUDA Functions of Multi-Scale Deformable Attention`

```shell
pip install MultiScaleDeformableAttention
```

## Training

To train on a single machine with 3 gpus for 50 epochs run:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2
python main.py --running_mode train --shard_id 0 --num_shards 1 --num_gpus 3 --cfg configs/ua_demmotr_t.yaml
```
