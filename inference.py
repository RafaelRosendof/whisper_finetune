# inference.py

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from datasets import Audio,load_from_disk,DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
from torch.utils.data import DataLoader
import jiwer

@dataclass
class DataCollatorSpeechSeq2SeqWithDynamicPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature[128]) for feature in features)
        batch = self.processor.feature_extractor.pad(features, return_tensors="pt", padding="True", max_length=max_length)
        if (batch["labels"][:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            batch["labels"] = batch["labels"][:, 1:]
        return batch
    
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="pt", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    labels = tokenizer(batch["sentence"]).input_ids
    return {"input_features": input_features, "labels": labels}


def main():
    common_voice = DatasetDict()
    common_voice["train"] = load_from_disk("/home/rafaelrosendo/whisper/dataset/common_voice/train")
    common_voice["test"] = load_from_disk("/home/rafaelrosendo/whisper/dataset/common_voice/test")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="pt", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="pt", task="transcribe")
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    common_voice = common_voice.map(
        prepare_dataset,
        remove_columns=common_voice.column_names["train"],
        num_proc=2
    )

    data_collator = DataCollatorSpeechSeq2SeqWithDynamicPadding(processor=processor)

    model = WhisperForConditionalGeneration.from_pretrained("/path/to/your/trained/model/directory")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    inference_dataloader = DataLoader(
        common_voice["test"],
        batch_size=2,  # Adjust batch size for inference
        collate_fn=data_collator,
        shuffle=False,
    )

    model.eval()
    for step, batch in enumerate(inference_dataloader):
        inputs = batch["input_features"].to(device)
        generated_ids = model.generate(inputs)

        # Process the generated output as needed

if __name__ == "__main__":
    main()
