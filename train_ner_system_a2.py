from datasets import load_dataset
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments
import torch


def modify_labels(ner_tags):
    reduced_labels = {
    "0": 0,
    "1": 12,
    "2": 12,
    "3": 11,
    "4": 11,
    "5": 8,
    "6": 8,
    "7": 1,
    "8": 1,
    "9": 2,
    "10": 2,
    "11": 3,
    "12": 3,
    "13": 4,
    "14": 4,
    "15": 5,
    "16": 5,
    "17": 6,
    "18": 6,
    "19": 7,
    "20": 7,
    "21": 9,
    "22": 9,
    "23": 10,
    "24": 10,
    "25": 13,
    "26": 13,
    "27": 14,
    "28": 14,
    "29": 15,
    "30": 15
    }
    tag = ner_tags['ner_tags']
    res = [reduced_labels[str(i)] for i in tag]
    ner_tags['ner_tags']=res
    return ner_tags


dataset = load_dataset("Babelscape/multinerd")
dataset = dataset.map(modify_labels, num_proc=8)

print(len(dataset['train']))

dataset = dataset.filter(lambda x: x['lang'] == 'en')
print(len(dataset['train']))

labels = ['ANIM', 'BIO', 'CEL', 'DIS', 'EVE', 'FOOD', 'INST', 
'LOC', 'MEDIA', 'MYTH', 'O', 'ORG', 'PER', 
'PLANT', 'TIME', 'VEHI']

model = SpanMarkerModel.from_pretrained('prajjwal1/bert-medium', ignore_mismatched_sizes=True, labels=labels)
print(model.config)
print(model)

args = TrainingArguments(
    output_dir="models/system_a2_test",
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
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
trainer.save_model("checkpoints/system_a2")
