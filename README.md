# rnnmorph
[![Current version on PyPI](http://img.shields.io/pypi/v/rnnmorph.svg)](https://pypi.python.org/pypi/rnnmorph)
[![Python versions](https://img.shields.io/pypi/pyversions/rnnmorph.svg)](https://pypi.python.org/pypi/rnnmorph)
[![Build Status](https://travis-ci.org/IlyaGusev/rnnmorph.svg?branch=master)](https://travis-ci.org/IlyaGusev/rnnmorph)
[![Code Climate](https://codeclimate.com/github/IlyaGusev/rnnmorph/badges/gpa.svg)](https://codeclimate.com/github/IlyaGusev/rnnmorph)

Morphological analyzer (POS tagger) for Russian and English languages based on neural networks and dictionary-lookup systems (pymorphy2, nltk).

### Русский язык, MorphoRuEval-2017 test dataset
Lenta:
* Качество по тегам:
  * 4025 меток из 4179, точность 96.31%
  * 4096 PoS тегов из 4179, точность 98.01%
  * 279 предложений из 358, точность 77.93%
* Качество полного разбора (включая леммы):
  * 3885 слов из 4179, точность 92.96%
  * 189 предложений из 358, точность 52.79%

VK:
* Качество по тегам:
  * 3691 меток из 3877, точность 95.20%
  * 3801 PoS тегов из 3877, точность 98.04%
  * 422 предложений из 568, точность 74.30%
* Качество полного разбора (включая леммы):
  * 3569 слов из 3877, точность 92.06%
  * 344 предложений из 568, точность 60.56%

JZ:
* Качество по тегам:
  * 3875 меток из 4042, точность 95.87%
  * 3990 PoS тегов из 4042, точность 98.71%
  * 288 предложений из 394, точность 73.10%
* Качество полного разбора (включая леммы):
  * 3656 слов из 4042, точность 90.45%
  * 170 предложений из 394, точность 43.15%

All:
* Точность по тегам по всем разделам: 95.81%
* Точность по PoS тегам по всем разделам: 98.26%
* Точность по предложениям по всем разделам: 74.92%

### English language, UD EWT test
* Only tags:
  * 13088 correct full tags of 14293, accuracy: 91.57%
  * 13449 correct PoS tags of 14293, accuracy: 94.10%
  * 1312 correct sentences of 2077, accuracy: 63.17%
* With lemmas:
  * 12438 correct words of 14293, accuracy: 87.02%
  * 1059 correct sentences of 2077, accuracy: 50.99%
  
Скорость: от 200 до 600 слов в секунду.

Потребление оперативной памяти: зависит от режима работы, для предсказания одиночных предложений - 500-600 Мб, для режима с батчами - пропорционально размеру батча.

### Install ###
```
sudo pip3 install rnnmorph
```
  
### Usage ###
```
from rnnmorph.predictor import RNNMorphPredictor
predictor = RNNMorphPredictor(language="ru")
forms = predictor.predict(["мама", "мыла", "раму"])
print(forms[0].pos)
>>> NOUN
print(forms[0].tag)
>>> Case=Nom|Gender=Fem|Number=Sing
print(forms[0].normal_form)
>>> мама
print(forms[0].vector)
>>> [0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 1 0 0 1]

```

### Acknowledgements ###
* Anastasyev D. G., Gusev I. O., Indenbom E. M., 2018, [Improving Part-of-speech Tagging Via Multi-task Learning and Character-level Word Representations](http://www.dialog-21.ru/media/4282/anastasyevdg.pdf)
* Anastasyev D. G., Andrianov A. I., Indenbom E. M., 2017, [Part-of-speech Tagging with Rich Language Description](http://www.dialog-21.ru/media/3895/anastasyevdgetal.pdf), [презентация](http://www.dialog-21.ru/media/4102/anastasyev.pdf)
* [Дорожка по морфологическому анализу "Диалога-2017"](http://www.dialog-21.ru/evaluation/2017/morphology/)
* [Материалы дорожки](https://github.com/dialogue-evaluation/morphoRuEval-2017)
* [Morphine by kmike](https://github.com/kmike/morphine), [CRF classifier for MorphoRuEval-2017 by kmike](https://github.com/kmike/dialog2017)
* [Universal Dependencies](http://universaldependencies.org/)
* Tobias Horsmann and Torsten Zesch, 2017, [Do LSTMs really work so well for PoS tagging? – A replication study](http://www.ltl.uni-due.de/wp-content/uploads/horsmannZesch_emnlp2017.pdf)
* Barbara Plank, Anders Søgaard, Yoav Goldberg, 2016, [Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss](https://arxiv.org/abs/1604.05529)
