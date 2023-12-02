print('Loading dependancies')
from datasets import load_dataset
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments, EarlyStoppingCallback
import torch

def modify_labels(ner_tags):
    required_ids = set([12,11,8,4,1])
    tag = ner_tags['ner_tags']
    res = [i if i in required_ids else 0 for i in tag]
    ner_tags['ner_tags']=res
    return ner_tags

print('Loading datasets')
dataset = load_dataset("Babelscape/multinerd")
print(len(dataset['train']))

dataset = dataset.filter(lambda x: x['lang'] == 'en')
dataset = dataset.map(modify_labels, num_proc=8)

print(len(dataset['train']))

model_a1 = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd")
model_a1.load_state_dict(torch.load('./checkpoints/system_a1/pytorch_model.bin'))
model_a1_state_dict = {k:v for k,v in model_a1.state_dict().items() if k not in ['classifier.bias', 'classifier.weight']}

model = SpanMarkerModel.from_pretrained('./pretrained_models/models--tomaarsen--span-marker-mbert-base-multinerd/snapshots/bfbb17381e16be9bce0c1f767a7a4708a8d12ca9/', ignore_mismatched_sizes=True)
model.load_state_dict(model_a1_state_dict, strict=False)
print(model)

args = TrainingArguments(
    output_dir="models/system_c1",
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
trainer.save_model("checkpoints/system_c1/")

