export CUDA_VISIBLE_DEVICES=1,2
export NCCL_BLOCKING_WAIT=1

python main.py --running_mode train --shard_id 0 --num_shards 1 --num_gpus 2 --cfg configs/ua_destmotr_train.yaml --opts resume data/demmotr_tiny_uadetrac_70e.pth save_period 1 eval_period 2 batch_size 4