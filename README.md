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
  * [7B](https://huggingface.co/IlyaGusev/llama_7b_ru_turbo_alpaca_lora)
  * [13B](https://huggingface.co/IlyaGusev/llama_13b_ru_turbo_alpaca_lora)
* Trained on Russian and English Alpacas

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
  * [7B](https://huggingface.co/IlyaGusev/saiga_7b_lora)
  * [13B](https://huggingface.co/IlyaGusev/saiga_13b_lora)
  * [30B](https://huggingface.co/IlyaGusev/saiga_30b_lora)
* Trained on 4 datasets: ru_turbo_saiga, ru_turbo_alpaca, ru_sharegpt_cleaned, oasst1_ru_main_branch
