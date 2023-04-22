#!/bin/bash
set -euo pipefail

# Saiga 30B
#python3 -m src.infer_saiga \
#    --model-name IlyaGusev/saiga_30b_lora \
#    --template-path internal_prompts/saiga_v2.json \
#    --input-path data/vicuna_question_ru.jsonl \
#    --output-path data/vicuna_saiga30b_answers.jsonl \
#    --batch-size 8

#python3 -m src.infer_saiga \
#    --model-name models/saiga_30b_lora \
#    --template-path internal_prompts/saiga_v2.json \
#    --input-path data/user_oriented_ru.jsonl \
#    --output-path data/user_saiga30b_answers.jsonl \
#    --batch-size 4

# Saiga 7B
python3 -m src.infer_saiga \
    --model-name IlyaGusev/saiga_7b_lora \
    --template-path internal_prompts/saiga_v2.json \
    --input-path data/vicuna_question_ru.jsonl \
    --output-path data/vicuna_saiga7b_answers.jsonl \
    --batch-size 8

python3 -m src.infer_saiga \
    --model-name IlyaGusev/saiga_7b_lora \
    --template-path internal_prompts/saiga_v2.json \
    --input-path data/user_oriented_ru.jsonl \
    --output-path data/user_saiga7b_answers.jsonl \
    --batch-size 8

# Saiga 13B
python3 -m src.infer_saiga \
    --model-name IlyaGusev/saiga_13b_lora \
    --template-path internal_prompts/saiga_v2.json \
    --input-path data/vicuna_question_ru.jsonl \
    --output-path data/vicuna_saiga13b_answers.jsonl \
    --batch-size 4

python3 -m src.infer_saiga \
    --model-name IlyaGusev/saiga_13b_lora \
    --template-path internal_prompts/saiga_v2.json \
    --input-path data/user_oriented_ru.jsonl \
    --output-path data/user_saiga13b_answers.jsonl \
    --batch-size 4

# Alpaca 13B
python3 -m src.infer_alpaca \
    --model-name IlyaGusev/llama_13b_ru_turbo_alpaca_lora \
    --template-path internal_prompts/ru_alpaca.json \
    --input-path data/vicuna_question_ru.jsonl \
    --output-path data/vicuna_rualpaca13b_answers.jsonl \
    --batch-size 1

python3 -m src.infer_alpaca \
    --model-name IlyaGusev/llama_13b_ru_turbo_alpaca_lora \
    --template-path internal_prompts/ru_alpaca.json \
    --input-path data/user_oriented_ru.jsonl \
    --output-path data/user_rualpaca13b_answers.jsonl \
    --batch-size 1

# Alpaca 7B
python3 -m src.infer_alpaca \
    --model-name IlyaGusev/llama_7b_ru_turbo_alpaca_lora \
    --template-path internal_prompts/ru_alpaca_old.json \
    --input-path data/vicuna_question_ru.jsonl \
    --output-path data/vicuna_rualpaca7b_answers.jsonl \
    --batch-size 1

python3 -m src.infer_alpaca \
    --model-name IlyaGusev/llama_7b_ru_turbo_alpaca_lora \
    --template-path internal_prompts/ru_alpaca_old.json \
    --input-path data/user_oriented_ru.jsonl \
    --output-path data/user_rualpaca7b_answers.jsonl \
    --batch-size 1
