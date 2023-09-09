import os
import re
import hashlib

from tqdm import tqdm
import fire
from kandinsky2 import get_kandinsky2
from transliterate import translit

from src.util.io import read_jsonl, write_jsonl


def generate_images(
    input_path,
    output_path,
    images_dir
):
    characters = read_jsonl(input_path)
    model = get_kandinsky2(
        'cuda',
        task_type='text2img',
        cache_dir='/tmp/kandinsky2',
        model_version='2.1',
        use_flash_attention=False
    )

    for char in tqdm(characters):
        prompt = char["image_prompt"]
        file_name = translit(char["name"], reversed=True).lower()
        file_name = re.sub(r'[^\w\s]', '', file_name)

        context_hash = hashlib.md5(char["context"].encode()).hexdigest()[:5]
        file_name = "_".join(file_name.split(" ")) + "_" + context_hash + ".png"
        file_path = os.path.join(images_dir, file_name)

        images = model.generate_text2img(
            prompt,
            num_steps=100,
            batch_size=1,
            guidance_scale=4,
            h=768, w=768,
            sampler='p_sampler',
            prior_cf_scale=4,
            prior_steps="5"
        )
        image = images[0]
        image.save(file_path, "PNG")
        char["image_path"] = file_path
    write_jsonl(characters, output_path)


if __name__ == "__main__":
    fire.Fire(generate_images)
