# Multimodal MultiObject Tracking Transformer

## Require

```shell
conda create -n mmotr python=3.9
conda activate mmotr

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

conda install -c conda-forge opencv matplotlib scipy pandas psutil tqdm simplejson einops pyyaml motmetrics

pip install tensorboard moviepy
```

- If the usage of deformableDETR, you should install `CUDA Functions of Multi-Scale Deformable Attention`

```shell
pip install MultiScaleDeformableAttention
```
