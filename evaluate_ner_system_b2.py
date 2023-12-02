from datasets import load_dataset
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments
import torch

# def reduce_labels(ner_tags):
#     reduced_labels = {
#         0: 0,
#         1: 5,
#         4: 4,
#         8: 1,
#         }
#     tag = ner_tags['ner_tags']
#     res = [reduced_labels[i] for i in tag]
#     ner_tags['ner_tags']=res
#     return ner_tags

def modify_labels(ner_tags):
    map = {
        0:0,
        1:1,
        4:2,
        8:3,
        11:4,
        12:5
    }
    required_ids = set([12,11,8,4,1])
    tag = ner_tags['ner_tags']
    res = [map[i] if i in required_ids else 0 for i in tag]
    ner_tags['ner_tags']=res
    return ner_tags

dataset = load_dataset("Babelscape/multinerd")
print(len(dataset['train']))

dataset = dataset.filter(lambda x: x['lang'] == 'en', num_proc=8)
dataset = dataset.map(modify_labels, num_proc=8)

print(len(dataset['train']))

labels = ['ANIM', 'DIS', 'LOC', 'O', 'ORG', 'PER']

model = SpanMarkerModel.from_pretrained('prajjwal1/bert-medium', ignore_mismatched_sizes=True, labels=labels)
print(model)

args = TrainingArguments(
    output_dir="models/system_b2",
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=args,
)

# Compute & save the metrics on the test set
trainer.model.load_state_dict(torch.load('checkpoints/system_b2/pytorch_model.bin'))
metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
print(metrics)