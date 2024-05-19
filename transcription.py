"""
Uses Speculative Decoding for faster inference
"""
from typing import Optional

import torch

from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def load_transcriber(
        model_id: str,
        assistant_model_id: Optional[str] = None
    ):
    # Model
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    if assistant_model_id is None:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )
    else:
        # Assistant
        assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        assistant_model.to(device)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            generate_kwargs={"assistant_model": assistant_model},
            torch_dtype=torch_dtype,
            device=device,
        )

    return pipe
