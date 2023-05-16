# rulm

Language models for Russian language: implementation and comparison.

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
* Training configs:
  * [self_instruct/configs/saiga_7b.json](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/saiga_7b.json)
  * [self_instruct/configs/saiga_13b.json](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/saiga_13b.json)
  * [self_instruct/configs/saiga_30b.json](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/saiga_30b.json)
* Base models:
  * [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
  * [huggyllama/llama-13b](https://huggingface.co/huggyllama/llama-13b)
  * [huggyllama/llama-30b](https://huggingface.co/huggyllama/llama-30b)
* Trained on 4 datasets: ru_turbo_saiga, ru_turbo_alpaca, ru_sharegpt_cleaned, oasst1_ru_main_branch. The script for smart merging: [create_chat_set.py](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/src/data_processing/create_chat_set.py)


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

## Questions v1
* User-oriented questions: https://github.com/IlyaGusev/rulm/blob/master/self_instruct/data/user_oriented_ru.jsonl
* Vicuna questions: https://github.com/IlyaGusev/rulm/blob/master/self_instruct/data/vicuna_question_ru.jsonl

### Toloka
* saiga30b vs saiga13b: 122-17-91
* saiga7b vs saiga13b: 97-11-122
* turbo vs rualpaca13b: 150-14-66
* turbo vs saiga30b: 121-12-97

### Telegram bot user_oriented/vicuna_questions (unfinished)
* rualpaca7b vs rualpaca13b: 30-46-58
* saiga30b vs davinci002: 92-30-18
* saiga30b vs saiga13b: 70-45-43
* saiga7b vs saiga13b: 42-53-54
* turbo vs davinci002: 91-24-15
* turbo vs fred: 120-10-8
* turbo vs rualpaca13b: 86-44-28
* turbo vs saiga30b: 60-46-52

## Questions v2
* User-oriented questions: https://github.com/IlyaGusev/rulm/blob/master/self_instruct/data/user_oriented_ru_v2.jsonl
* Vicuna questions: https://github.com/IlyaGusev/rulm/blob/master/self_instruct/data/vicuna_question_ru.jsonl

### Toloka
* turbo vs gpt4: 46-8-122
* turbo vs saiga30b: 111-9-56
* turbo vs saiga30bq4_1: 121-9-46

## RSG
* RussianSuperGLUE: [link](https://russiansuperglue.com/leaderboard/2)
* Zero-shot mode

<img width="511" alt="изображение" src="https://user-images.githubusercontent.com/2670295/236706862-1d2f27fe-0cf8-4858-87e7-b6392aaa659c.png">





