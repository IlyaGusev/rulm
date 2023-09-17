import os
import random

import torch
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _check_candidates(candidates, bad_ids, tokenizer, backup_token):
    for token_id in candidates:
        if token_id not in bad_ids:
            token = tokenizer.convert_ids_to_tokens(token_id)
            return token_id, token
    return None, backup_token


def fix_tokenizer(tokenizer, model_config):
    bad_ids = (None, tokenizer.vocab_size)

    special_tokens = dict()
    guessed_pad_token_id = None
    guessed_bos_token_id = None
    guessed_eos_token_id = None
    guessed_unk_token_id = None
    for token_id in range(1000):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if tokenizer.pad_token_id in bad_ids and guessed_pad_token_id is None and "pad" in token:
            guessed_pad_token_id = token_id
        if tokenizer.bos_token_id in bad_ids and guessed_bos_token_id is None and "<s>" in token:
            guessed_bos_token_id = token_id
        if tokenizer.eos_token_id in bad_ids and guessed_eos_token_id is None and "</s>" in token:
            guessed_eos_token_id = token_id
        if tokenizer.unk_token_id in bad_ids and guessed_unk_token_id is None and "unk" in token:
            guessed_unk_token_id = token_id

    if tokenizer.pad_token_id in bad_ids:
        candidates = (
            model_config.pad_token_id,
            guessed_pad_token_id,
            tokenizer.unk_token_id
        )
        token_id, token = _check_candidates(candidates, bad_ids, tokenizer, "<pad>")
        tokenizer.pad_token_id = token_id
        special_tokens["pad_token"] = token

    if tokenizer.bos_token_id in bad_ids:
        candidates = (
            model_config.bos_token_id,
            guessed_bos_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.eos_token_id,
        )
        token_id, token = _check_candidates(candidates, bad_ids, tokenizer, "<s>")
        tokenizer.bos_token_id = token_id
        special_tokens["bos_token"] = token

    if tokenizer.eos_token_id in bad_ids:
        candidates = (
            model_config.eos_token_id,
            guessed_eos_token_id,
            tokenizer.bos_token_id
        )
        token_id, token = _check_candidates(candidates, bad_ids, tokenizer, "</s>")
        tokenizer.eos_token_id = token_id
        special_tokens["eos_token"] = token

    if tokenizer.unk_token_id in bad_ids:
        candidates = (
            model_config.unk_token_id,
            guessed_unk_token_id
        )
        token_id, token = check_candidates(candidates, bad_ids, tokenizer, "<unk>")
        tokenizer.unk_token_id = token_id
        special_tokens["unk_token"] = token

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.padding_side = "left"
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    if hasattr(model_config, "n_positions"):
        n_positions = getattr(model_config, "n_positions")
        if n_positions:
            tokenizer.model_max_length = n_positions
    if hasattr(model_config, "max_position_embeddings"):
        max_position_embeddings = getattr(model_config, "max_position_embeddings")
        if max_position_embeddings:
            tokenizer.model_max_length = max_position_embeddings

    print("Vocab size: ", tokenizer.vocab_size)
    print("PAD: ", tokenizer.pad_token_id, tokenizer.pad_token)
    print("BOS: ", tokenizer.bos_token_id, tokenizer.bos_token)
    print("EOS: ", tokenizer.eos_token_id, tokenizer.eos_token)
    print("UNK: ", tokenizer.unk_token_id, tokenizer.unk_token)
    print("SEP: ", tokenizer.sep_token_id, tokenizer.sep_token)
    return tokenizer


def fix_model(model, tokenizer, use_resize=True):
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.pad_token_id is not None

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.decoder_start_token_id = model.config.bos_token_id
    assert model.config.bos_token_id is not None

    model.config.eos_token_id = tokenizer.eos_token_id
    assert model.config.eos_token_id is not None

    model.config.unk_token_id = tokenizer.unk_token_id
    assert model.config.unk_token_id is not None

    if use_resize:
        model.resize_token_embeddings(len(tokenizer))

    return model


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch
