# nlp_project

Git clone llama factory[https://github.com/hiyouga/LLaMA-Factory] first.

* vllm

  ```shell
  conda activate llava
  bash infer_llama3_vllm.sh
  python infer_llama3_vllm.py
  ```

* langchain

  ```shell
  conda activate llava
  bash infer_llama3_langchain.sh
  python infer_llama3_langchain.py
  ```



Details for infer_llama3_vllm.sh

```
CUDA_VISIBLE_DEVICES=4 API_PORT=8001 llamafactory-cli api /home/ubuntu/6000N/llama_factory/LLaMA-Factory-main/examples/inference/llama3_vllm.yaml
```

* Set APT_PORT as you like, but remember to change the port in python script.
* Revise the parameter setting in *.yaml