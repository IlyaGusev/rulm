## Training

Overview:

* Tested with `nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04` Docker image
* Install dependencies. You will need Git LFS to download the model and a **correct combination** of the versions of `transformers`, `peft`, and `bitsandbytes`. `wandb` is optional.
* Download a base model that you will be finetuning, for example, [TheBloke/Llama-2-7B-fp16](https://huggingface.co/TheBloke/Llama-2-7B-fp16).
* Fix treatment of `pad`, `bos`, `eos` tokens.
* Prepare your data as two JSONL files, with three fields for the `"instruct"` mode: `"instruction"`, `"input"`, `"output"`. Or the following fields for the `"chat"` mode: `"messages"`.
* Run training.

### Install libraries
```
sudo apt-get install git-lfs
pip install -r ../requirements.txt
```

### Download base model
```
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="TheBloke/Llama-2-7B-fp16", local_dir="models/llama2-7b", ignore_patterns=["LICENSE", "README.md", "*.safetensors"])'
```

or for new versions of Transformers (and if there are *.safetensors files):

```
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="TheBloke/Llama-2-7B-fp16", local_dir="models/llama2-7b", ignore_patterns=["LICENSE", "README.md", "*.bin"])'
```


### Fix tokenizer
Edit the following files under `models/llama-7b`:

`tokenizer_config.json`:

```
{
    "tokenizer_class": "LlamaTokenizer",
    "model_max_length": 4096,
    "padding_side": "left",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<unk>",
    "unk_token": "<unk>",
    "clean_up_tokenization_spaces": false,
    "special_tokens_map_file": "special_tokens_map.json"
}
```

`special_tokens_map.json`:

```
{
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<unk>",
    "unk_token": "<unk>"
}
```

### Prepare data

Create two JSONL files with training and validation sets. See [create_short_chat_set.py](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/src/data_processing/create_short_chat_set.py) for an example.

### Run training
```python
python3 -m src.train --config-file configs/saiga2_7b.json --train-file train.jsonl --val-file val.jsonl  --output-dir models/saiga2_7b --omit-base-model-save
```
