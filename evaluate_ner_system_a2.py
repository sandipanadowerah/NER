from datasets import load_dataset
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArgument
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
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=args,
)

trainer.model.load_state_dict(torch.load('checkpoints/system_a2/pytorch_model.bin'))
# Compute & save the metrics on the test set
metrics = trainer.evaluate(dataset['test'], metric_key_prefix="test")
print(metrics)