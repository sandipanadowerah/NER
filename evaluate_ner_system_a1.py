print('Loading dependancies')
from datasets import load_dataset
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments, EarlyStoppingCallback
import torch

print('Loading datasets')
dataset = load_dataset("Babelscape/multinerd")
print(len(dataset['train']))

dataset = dataset.filter(lambda x: x['lang'] == 'en')
print(len(dataset['train']))

model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd")
print(model)

args = TrainingArguments(
    output_dir="models/system_a1",
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=args,
)

# Compute & save the metrics on the test set
trainer.model.load_state_dict(torch.load('checkpoints/system_a1/pytorch_model.bin'))
metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
print(metrics)