# training.py
import tokenize
from datasets import DatasetDict,load_dataset,Audio,load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tokenizers import Tokenizer
import tokenizers
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import DataLoader
import jiwer
import evaluate
from torch.nn.utils.rnn import pad_sequence

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="pt", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="pt", task="transcribe")

@dataclass
class DataCollatorSpeechSeq2SeqWithDynamicPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract the tensors from the '128' key in each feature dictionary
        tensors = [torch.tensor(feature.get(128, [])) for feature in features]

        # Pad the sequences to the same length within the batch
        padded_tensors = pad_sequence(tensors, batch_first=True, padding_value=0)

        # Pad using the feature extractor
        batch = self.processor.feature_extractor.pad(
            {"input_values": padded_tensors},  # Assuming 'input_values' is the key for your input tensors
            return_tensors="pt",
            padding=True,
        )

        if (batch["labels"][:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            batch["labels"] = batch["labels"][:, 1:]

        return batch



def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenize.pad_token_id
    pred_str = tokenizers.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer_metric = jiwer.WER()
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    print(f"WER: {wer}")
    return {"wer": wer}

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

    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2 )

    data_collator = DataCollatorSpeechSeq2SeqWithDynamicPadding(processor=processor)
    metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir="/home/rafaelrosendo/whisper/saida",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=100,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        num_train_epochs=5
    )

    trainer_1 = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer_1.train()
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
