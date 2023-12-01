# training.py
import tokenize
from datasets import DatasetDict,load_dataset,Audio,load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tokenizers import Tokenizer
import tokenizers
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, BatchFeature
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import DataLoader
import jiwer
import evaluate
from torch.nn.utils.rnn import pad_sequence

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="pt", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="pt", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    labels = tokenizer(batch["sentence"]).input_ids
    return {"input_features": input_features, "labels": labels}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    common_voice = DatasetDict()
    common_voice["train"] = load_from_disk("/home/rafaelrosendo/whisper/dataset/common_voice/train")
    common_voice["test"] = load_from_disk("/home/rafaelrosendo/whisper/dataset/common_voice/test")


    print("common_voice")    
    print(common_voice["train"][0])
    print(common_voice["train"][0]["sentence"])

    print(type(common_voice["train"]))


    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")
    tprocessor = WhisperProcessor.from_pretrained("openai/whisper-large", language="pt", task="transcribe")
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="pt", task="transcribe")
   # common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2 )
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2 )
    
    data_train = DataLoader(
    common_voice["train"].dataset,
    batch_size=4,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=2,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,  
    prefetch_factor=2,
    persistent_workers=False,
)
    metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir="/home/rafaelrosendo/whisper/saida",
        per_device_train_batch_size=8,
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
        num_train_epochs=2
    )

    trainer_1 = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset= data_train,
        eval_dataset=common_voice["test"],
        tokenizer=processor.feature_extractor,
    )

    trainer_1.train()
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()


'''
Traceback (most recent call last):
  File "/home/rafaelrosendo/whisper/rrr.py", line 127, in <module>
  File "/home/rafaelrosendo/whisper/rrr.py", line 123, in main
  File "/home/rafaelrosendo/anaconda3/envs/torch2/lib/python3.9/site-packages/transformers/trainer.py", line 1555, in train
    return inner_training_loop(
  File "/home/rafaelrosendo/anaconda3/envs/torch2/lib/python3.9/site-packages/transformers/trainer.py", line 1838, in _inner_training_loop
    for step, inputs in enumerate(epoch_iterator):
  File "/home/rafaelrosendo/anaconda3/envs/torch2/lib/python3.9/site-packages/accelerate/data_loader.py", line 384, in __iter__
    current_batch = next(dataloader_iter)
  File "/home/rafaelrosendo/anaconda3/envs/torch2/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 633, in __next__
    data = self._next_data()
  File "/home/rafaelrosendo/anaconda3/envs/torch2/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 677, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/home/rafaelrosendo/anaconda3/envs/torch2/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 54, in fetch
    return self.collate_fn(data)
  File "/home/rafaelrosendo/whisper/rrr.py", line 38, in __call__
    
  File "/home/rafaelrosendo/anaconda3/envs/torch2/lib/python3.9/site-packages/transformers/feature_extraction_sequence_utils.py", line 132, in pad
    raise ValueError(
ValueError: You should supply an instance of `transformers.BatchFeature` or list of `transformers.BatchFeature` to this method that includes input_features, but you provided ['input_values']
  0%|                                                                                           | 0/4000 [00:00<?, ?it/s]
'''