# nlp_project

## environment
```shell
conda create -n llama python=3.12.0
conda activate llama
pip install -r requirements.txt
```

## baseline inference
We utilize vllm to implement LLaMA3.1-8B
```shell
conda activate llama
bash infer_llama3_vllm.sh
python infer_llama3_vllm.py
```


Details for infer_llama3_vllm.sh
```
CUDA_VISIBLE_DEVICES=4 API_PORT=8001 llamafactory-cli api /home/ubuntu/6000N/llama_factory/LLaMA-Factory-main/examples/inference/llama3_vllm.yaml
```

* Download the checkpoints of LLaMA3.1-8B first
* Set APT_PORT as you like, but remember to change the port in python script.
* Revise the parameter setting in *.yaml


## LoRA
We utilize LLaMA-Factory to conduct LoRA experiments.
```shell
conda activate llama
bash lora_llama3_ds.sh
```

Details for lora_llama3_ds.sh
```
NNODES=1
FORCE_TORCHRUN=1
MASTER_PORT=29501
NPROC_PER_NODE=4
RANK=0
MASTER_ADDR=0.0.0.0

############ LLama3 ############
config_yaml=llama3_lora_sft.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py /home/ubuntu/6000N/llama_factory/LLaMA-Factory-main/examples/train_lora/$config_yaml
```

* Set NPROC_PER_NODE, MASTER_PORT and CUDA_VISIBLE_DEVICES as you like.
* Revise the parameter setting in *.yaml