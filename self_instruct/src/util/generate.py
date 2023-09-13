from typing import List
from transformers import AutoModel, AutoTokenizer, GenerationConfig


def generate(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    generation_config: GenerationConfig,
    source_max_length: int = 512
):
    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=source_max_length,
        padding=True
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )
    outputs = []
    for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
        sample_output_ids = sample_output_ids[len(sample_input_ids):]
        sample_output = tokenizer.decode(sample_output_ids, skip_special_tokens=True)
        sample_output = sample_output.replace("</s>", "").strip()
        outputs.append(sample_output)
    return outputs


