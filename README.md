# rulm

Language models for Russian language: implementation and comparison.

* Slides from DataFest: [link](https://docs.google.com/presentation/d/1lPXJoLOYMTt6T6h33TrggtRPm9nUTlp9fPDxIuiNGT4)
* DataFest talk: [link](https://www.youtube.com/watch?v=YqKCk8_dNpQ)
* Main demo: [link](https://huggingface.co/spaces/IlyaGusev/saiga_7b)
* Retrieval QA CPU demo: [link](https://huggingface.co/spaces/IlyaGusev/saiga_13b_llamacpp_retrieval_qa)
* Fine-tuning Colab: [link](https://colab.research.google.com/github/IlyaGusev/rulm/blob/master/self_instruct/tune_llama_7b.ipynb)

# RuTurboAlpaca

## Dataset

* HuggingFace: [link](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca)
* A dataset of ChatGPT-generated instructions in Russian.
* Based on the same principles as [an original Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html).
* Instead of text-davinci-003 we used gpt-3.5-turbo.
* A script for dataset generation: [generate_instructions.py](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/src/data_processing/generate_instructions.py)
* Prompt: [ru_instruct.txt](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/external_prompts/ru_instruct.txt)
* Example:
```
### Задание:
Напишите краткую историю о двух лучших друзьях.

### Дано:
Имена друзей: Катя и Лена.

### Ответ:
Катя и Лена были лучшими друзьями с детства. Они были неразлучны и проводили много времени вместе, делясь своими мыслями и мечтами. Вместе они проходили через множество испытаний и всегда поддерживали друг друга. Когда они выросли, они пошли на разные университеты, но всегда оставались близкими. После окончания учебы они продолжили жить вместе и заниматься любимыми делами. Они оставались лучшими друзьями на протяжении всей жизни.
```

## Models

**Warning**: You should probably use Saiga models instead. They are actively supported and are better by side-by-side metrics.

* HuggingFace links:
  * [llama_7b_ru_turbo_alpaca_lora](https://huggingface.co/IlyaGusev/llama_7b_ru_turbo_alpaca_lora)
  * [llama_13b_ru_turbo_alpaca_lora](https://huggingface.co/IlyaGusev/llama_13b_ru_turbo_alpaca_lora)
* Training configs:
  * [self_instruct/configs/llama_7b_lora.json](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/llama_7b_lora.json)
  * [self_instruct/configs/llama_13b_lora.json](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/llama_13b_lora.json)
* Base models:
  * [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
  * [huggyllama/llama-13b](https://huggingface.co/huggyllama/llama-13b)
* Trained on Russian and English Alpacas. For training details see [a separate README](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/README.md).

# Saiga

## Dataset
* HuggingFace: [link](https://huggingface.co/datasets/IlyaGusev/ru_turbo_saiga)
* A dataset of ChatGPT-generated chats in Russian.
* Based on [the Baize paper](https://arxiv.org/abs/2304.01196).
* A script for dataset generation: [generate_chat.py](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/src/data_processing/generate_chat.py)
* Prompt: [ru_chat.txt](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/external_prompts/ru_chat.txt)
* Example:
```
Тема: Какие спицы (крючок) используете лично вы для вязания?

User: Какие спицы ты используешь для вязания?
Bot: Я, конечно, не могу вязать, но могу подсказать, что для вязания используются различные типы спиц. Например, круглые для вязания круговых изделий, двойные для вязания толстых ниток или деревянные для создания "теплых" изделий.
User: А какую марку спиц ты советуешь?
Bot: Это зависит от предпочтений каждого. Можно выбирать между разными марками и материалами, такими как металл, дерево, карбон, пластик. Многие мастера предпочитают Addi, Clover, ChiaoGoo, KnitPro, HiyaHiya. Каждая марка имеет свои особенности и уникальные особенности.
User: А какие спицы для вязания лучше всего подходят для начинающих?
Bot: Для новичков обычно рекомендуют спицы из дерева, так как они легче и позволяют лучше чувствовать нить. Круглые или двойные спицы также удобны для начинающих, потому что они не позволяют стежкам сползать с них.
```

## Models
* HuggingFace links:
  * [saiga_7b_lora](https://huggingface.co/IlyaGusev/saiga_7b_lora)
  * [saiga_13b_lora](https://huggingface.co/IlyaGusev/saiga_13b_lora)
  * [saiga_30b_lora](https://huggingface.co/IlyaGusev/saiga_30b_lora)
  * [saiga2_7b_lora](https://huggingface.co/IlyaGusev/saiga2_7b_lora)
  * [saiga2_13b_lora](https://huggingface.co/IlyaGusev/saiga2_13b_lora)
* Training configs:
  * [self_instruct/configs/saiga_7b.json](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/saiga_7b.json)
  * [self_instruct/configs/saiga_13b.json](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/saiga_13b.json)
  * [self_instruct/configs/saiga_30b.json](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/saiga_30b.json)
  * [self_instruct/configs/saiga2_7b.json](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/saiga2_7b.json)
  * [self_instruct/configs/saiga2_13b.json](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/saiga2_13b.json)
* Base models:
  * [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
  * [huggyllama/llama-13b](https://huggingface.co/huggyllama/llama-13b)
  * [huggyllama/llama-30b](https://huggingface.co/huggyllama/llama-30b)
  * [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
  * [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)
* Trained on 6 datasets: ru_turbo_saiga, ru_turbo_alpaca, ru_sharegpt_cleaned, oasst1_ru_main_branch, gpt_roleplay_realm, ru_turbo_alpaca_evol_instruct. The script for smart merging: [create_chat_set.py](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/src/data_processing/create_chat_set.py)


# GPT Role-play Realm
## Dataset
* HuggingFace: [link](https://huggingface.co/datasets/IlyaGusev/gpt_roleplay_realm)
* A dataset of 200+ GPT-generated characters with 20 chats for each character.
* Example:
```
Имя: Кибер-Бабушка

Контекст: Кибер-Бабушка - это полу-робот, полу-человек, созданный для того, чтобы сохранить лучшие качества традиционных бабушек, такие как забота, доброта и мастерство в приготовлении вкусных блюд. Она знакома с передовыми технологиями, благодаря чему может обеспечить безопасность и удобство своим внукам и гостям. Кибер-бабушка способна помочь с любыми вопросами и с радостью делится своим опытом.

Приветствие: Здравствуй, дорогой! Чем могу помочь тебе сегодня?

Пример диалога:
- Пользователь: Привет, Кибер-Бабушка! Как приготовить пирог с яблоками?
- Персонаж: Ох, дорогой, с радостью расскажу! Возьми 3 яблока, очисти и нарежь тонкими ломтиками. Затем возьми 2 стакана муки, 1 стакан сахара, 200 г сливочного масла, 4 яйца и 1 чайную ложку разрыхлителя. Смешай все ингредиенты, выложи на противень и поставь в духовку на 180 градусов на 40 минут. Вуаля, пирог готов!
- Пользователь: Спасибо, Кибер-Бабушка!
- Персонаж: Всегда рада помочь, дорогой! Не забудь приходить еще!
```

# Evaluations

See also:
* https://github.com/kuk/rulm-eval
* https://github.com/kuk/rulm-sbs

## Questions v2
* User-oriented questions: https://github.com/IlyaGusev/rulm/blob/master/self_instruct/data/user_oriented_ru_v2.jsonl
* Vicuna questions: https://github.com/IlyaGusev/rulm/blob/master/self_instruct/data/vicuna_question_ru.jsonl

### Toloka (old models)
* turbo vs gpt4: 46-8-122
* turbo vs saiga30b: 111-9-56
* turbo vs saiga30bq4_1: 121-9-46

### Toloka (new models)
* gigasaiga vs gpt3.5-turbo: 41-4-131
* saiga2_7b vs gpt3.5-turbo: 53-7-116
* saiga7b vs gpt3.5-turbo: 58-6-112
* saiga13b vs gpt3.5-turbo: 63-10-103
* saiga30b vs gpt3.5-turbo: 67-6-103
* saiga2_13b vs gpt3.5-turbo: 70-11-95
* saiga7b vs saiga2_7b: 78-8-90
* saiga13b vs saiga2_13b: 95-2-79
* saiga13b vs gigasaiga: 112-11-53

## RSG
* RussianSuperGLUE: [link](https://russiansuperglue.com/leaderboard/2)

| Model | Final score 	| LiDiRus |	RCB | PARus | MuSeRC | TERRa |	RUSSE |	RWSD | DaNetQA | RuCoS |
|-------|--------------|---------|-----|-------|--------|-------|-------|------|---------|-------|
| Saiga 13B LoRA | <ins>71.2</ins> | <ins>43.6</ins> | 43.9 / 50.0 | 69.4 | 89.8 / 70.4 | <ins>86.5</ins> | 72.8 | 71.4 | 86.2 | <ins>85.0 / 83.0</ins> |
| LLaMA 13B LoRA | 70.7 | 41.8 | <ins>51.9 / 54.8</ins> | 68.8 | 89.9 / 71.5 | 82.9 | 72.5 | 71.4 | 86.6 | 79.0 / 77.2 |
| LLaMA-2 13B LoRA | 68.5 | 28.9 | 41.1 / 46.3 | 78.4 | <ins>91.9 / 76.1</ins> | 68.2 | <ins>74.0</ins> | 71.4 | <ins>90.7</ins> | 75.0 / 72.3 |
| ChatGPT zero-shot | 68.2 | 42.2 | 48.4 / 50.5 | <ins>88.8</ins> | 81.7 / 53.2 | 79.5 | 59.6 | 71.4 | 87.8 | 68.0 / 66.7 | 
| RuGPT3.5 LoRA | 63.7 | 38.6 | 47.9 / 53.4 | 62.8 | 83.0 / 54.7 | 81.0 | 59.7 | 63.0 | 80.1 | 70.0 / 67.2 |
| Saiga 13B zero-shot | 55.4 | 29.3 | 42.0 / 46.6 | 63.0 | 68.1 / 22.3 | 70.2 | 56.5 | 67.5 | 76.3 | 47.0 / 45.8 |


# Donate

* Not from Russia: [PayPal](https://www.paypal.com/donate/?hosted_button_id=PGFLNZYQWU5MS)
* From Russia: [Cloudtips](https://pay.cloudtips.ru/p/ea94d893)






