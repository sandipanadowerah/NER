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
    # Training Hyperparameters:
    learning_rate=1e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,  # Replace `bf16` with `fp16` if your hardware can't use bf16.
    # Other Training parameters
    logging_steps=50,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_total_limit=1,
    dataloader_num_workers=4,
    metric_for_best_model='eval_loss',
    load_best_model_at_end = True,
    save_steps=1000,
    length_column_name = 'input_length',
    report_to='none',

)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],

)

trainer.train()
trainer.save_model("checkpoints/system_a1/")
