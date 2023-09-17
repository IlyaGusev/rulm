from transformers import AutoTokenizer, AutoConfig

from src.util.dl import fix_tokenizer


def test_fix_tokenizer():
    model_name = "TheBloke/Llama-2-7B-fp16"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = fix_tokenizer(tokenizer, config)
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 0
    assert tokenizer.model_max_length == 4096

    model_name = "ai-forever/ruGPT-3.5-13B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = fix_tokenizer(tokenizer, config)
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert tokenizer.unk_token_id == 1
    assert tokenizer.model_max_length == 2048
