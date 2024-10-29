NNODES=1
FORCE_TORCHRUN=1
MASTER_PORT=29501
NPROC_PER_NODE=4
RANK=0
MASTER_ADDR=0.0.0.0

############ LLama3 ############
#config_yaml=llama3_lora_sft.yaml
#config_yaml=llama3_lora_dpo.yaml
config_yaml=llama3_lora_sft.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py /home/ubuntu/6000N/llama_factory/LLaMA-Factory-main/examples/train_lora/$config_yaml
