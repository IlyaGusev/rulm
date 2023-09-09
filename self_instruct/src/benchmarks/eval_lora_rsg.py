from pathlib import Path
from typing import Tuple
import random
import fire

from datasets import load_dataset

from src.eval_rsg import (
    predict_danetqa,
    predict_rcb,
    predict_terra,
    predict_lidirus,
    predict_parus,
    predict_muserc,
    predict_rucos,
    predict_rwsd,
    predict_russe,
    clean_rwsd_response,
    ALL_TASKS,
    predict_saiga_zero_shot
)
from src.data_processing.convert_rsg import (
    DANETQA_SOURCE_TEMPLATE,
    RCB_SOURCE_TEMPLATE,
    TERRA_SOURCE_TEMPLATE,
    LIDIRUS_SOURCE_TEMPLATE,
    PARUS_CAUSE_SOURCE_TEMPLATE,
    PARUS_EFFECT_SOURCE_TEMPLATE,
    MUSERC_SOURCE_TEMPLATE,
    RUCOS_SOURCE_TEMPLATE,
    RWSD_SOURCE_TEMPLATE,
    RUSSE_SOURCE_TEMPLATE
)
from src.util.io import write_jsonl
from src.util.load import load_saiga

HF_DATASET = "RussianNLP/russian_super_glue"


def clean_danetqa(response):
    return "да" in response.lower()


def clean_rcb(response):
    if "да" in response.lower():
        return "entailment"
    if "нет" in response.lower():
        return "contradiction"
    return "neutral"


def clean_terra(response):
    if "да" in response.lower():
        return "entailment"
    return "not_entailment"


def clean_muserc(response):
    if "да" in response.lower():
        return True
    return False


def clean_rucos(response, entities):
    _ = entities
    return response


def clean_russe(response):
    if "да" in response.lower():
        return 1
    return 0


def main(
    model_name,
    nrows: int = None,
    template_path: str = "internal_prompts/saiga_v2.json",
    split: str = "test",
    predictions_dir: str = "submission_peft",
    debug: bool = False,
    tasks: Tuple[str] = ALL_TASKS
):
    predictions_dir = Path(predictions_dir)

    model, tokenizer, generation_config = load_saiga(model_name)
    generation_config.no_repeat_ngram_size = 64
    generation_config.temperature = 0.01

    def predict_saiga_zero_shot_bound(batch):
        generation_config.max_new_tokens = 256
        return predict_saiga_zero_shot(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            template_path=template_path,
            prompts=batch,
            debug=debug
        )

    if "danetqa" in tasks:
        predict_danetqa(
            split=split,
            predict_func=predict_saiga_zero_shot_bound,
            output_path=predictions_dir / "DaNetQA.jsonl",
            nrows=nrows,
            clean_func=clean_danetqa,
            template="Задание: danetqa\n" + DANETQA_SOURCE_TEMPLATE
        )
    if "rcb" in tasks:
        predict_rcb(
            split=split,
            predict_func=predict_saiga_zero_shot_bound,
            output_path=predictions_dir / "RCB.jsonl",
            nrows=nrows,
            clean_func=clean_rcb,
            template="Задание: rcb\n" + RCB_SOURCE_TEMPLATE
        )
    if "terra" in tasks:
        predict_terra(
            split=split,
            predict_func=predict_saiga_zero_shot_bound,
            output_path=predictions_dir / "TERRa.jsonl",
            nrows=nrows,
            clean_func=clean_terra,
            template="Задание: terra\n" + TERRA_SOURCE_TEMPLATE
        )
    if "lidirus" in tasks:
        predict_lidirus(
            predict_func=predict_saiga_zero_shot_bound,
            output_path=predictions_dir / "LiDiRus.jsonl",
            nrows=nrows,
            clean_func=clean_terra,
            template="Задание: terra\n" + LIDIRUS_SOURCE_TEMPLATE
        )
    if "parus" in tasks:
        predict_parus(
            split=split,
            predict_func=predict_saiga_zero_shot_bound,
            output_path=predictions_dir / "PARus.jsonl",
            nrows=nrows,
            template_cause="Задание: parus\n" + PARUS_CAUSE_SOURCE_TEMPLATE,
            template_effect="Задание: parus\n" + PARUS_EFFECT_SOURCE_TEMPLATE
        )
    if "muserc" in tasks:
        predict_muserc(
            split=split,
            predict_func=predict_saiga_zero_shot_bound,
            output_path=predictions_dir / "MuSeRC.jsonl",
            nrows=nrows,
            template="Задание: muserc\n" + MUSERC_SOURCE_TEMPLATE,
            clean_func=clean_muserc
        )
    if "rucos" in tasks:
        predict_rucos(
            split=split,
            predict_func=predict_saiga_zero_shot_bound,
            output_path=predictions_dir / "RuCoS.jsonl",
            nrows=nrows,
            template="Задание: rucos\n" + RUCOS_SOURCE_TEMPLATE,
            clean_func=clean_rucos
        )

    if "rwsd" in tasks:
        predict_rwsd(
            split=split,
            predict_func=predict_saiga_zero_shot_bound,
            output_path=predictions_dir / "RWSD.jsonl",
            nrows=nrows,
            template="Задание: rwsd\n" + RWSD_SOURCE_TEMPLATE,
            clean_func=clean_rwsd_response
        )
    if "russe" in tasks:
        predict_russe(
            split=split,
            predict_func=predict_saiga_zero_shot_bound,
            output_path=predictions_dir / "RUSSE.jsonl",
            nrows=nrows,
            template="Задание: russe\n" + RUSSE_SOURCE_TEMPLATE,
            clean_func=clean_russe
        )

if __name__ == "__main__":
    fire.Fire(main)
