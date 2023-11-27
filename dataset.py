from datasets import DatasetDict
from datasets import load_dataset, DatasetDict
from datasets import load_from_disk

# Load datasets
common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="train+validation")
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="test")

# Remove unnecessary variables present in the columns of the dataset
columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]

# Remove columns
for split in common_voice.keys():
    common_voice[split] = common_voice[split].remove_columns(columns_to_remove)

# Save datasets to disk
save_path = "/home/rafaelrosendo/whisper/dataset/common_voice"
common_voice.save_to_disk(save_path)

#/home/rafaelrosendo/whisper/dataset/common_voice/test
#/home/rafaelrosendo/whisper/dataset/common_voice/train