# NER
Assignment: Research Engineer in Natural Language Processing (Named entity recognition)
# NER Analysis for Transformer Models

**Dataset Used** 
MultiNERD dataset: Multilingual NER dataset covering ten languages. Link: https://huggingface.co/datasets/Babelscape/multinerd

**Pretrained models used**

2 different models have been trained in 6 different settings on the English subset of the MultiNERD dataset. Details are as follows:

Model 1: https://huggingface.co/tomaarsen/span-marker-mbert-base-multinerd (disk size: 712MB) (This model is already trained on all the languages of the MultiNERD dataset covering all the (15 + 1) classes).
Model 2: https://huggingface.co/prajjwal1/bert-medium (disk size: 167MB)

List of systems
1) System A1 :- Model 1 finetuned for all the 16 (15 + 1) classes of the dataset.
2) System A2 :- Model 2 finetuned for all the 16 (15 + 1) classes of the dataset.

3) System B1 :- Model 1 finetuned for only 6  (5+1) classes of the total 16: [PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM) OTHER (O)]
4) System B2 :- Model 2 finetuned for only 6 (5+1) classes of the total 16: [PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM) OTHER (O)]

5) System C1 :- System A1 finetuned for only 6 (5+1) classes of the total 16: [PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM) OTHER (O)].
6) System C2 :- System A2 finetuned for only 6 (5+1) classes of the total 16: [PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM) OTHER (O)].

## Usage 

* Install dependencies

    `pip install -r requirements.txt`
*   Run training code for the desired system directly  
    Ex:    `python train_ner_system_a1.py`
*   Extract models.zip in the main directory from the link, https://drive.google.com/file/d/1fRmXD-o8qfyrepXr6JJJhacyJpW30Amv/view?usp=sharing  
    Ex `unzip models.zip`
*   Similarly, we can run an evaluation code for the finetuned checkpoints. The script will print out evaluation metrics 
    Ex `python evaluate_ner_system_a1.py`


## Results
| System | F1 Score | Precision | Recall | Accuracy |
|--------|----------|-----------|--------|----------|
| A1     | 0.95097  | 0.94695   | 0.95503| 0.99019  |
| A2     | 0.92557  | 0.92467   | 0.92647| 0.96181  |
| B1     | 0.96799  | 0.95980   | 0.97632| 0.99810  |
| B2     | 0.96211  | 0.96272   | 0.96151| 0.99031  |
| C1     | 0.96597  | 0.95517   | 0.97701| 0.99797  |
| C2     | 0.96506  | 0.95941   | 0.97079| 0.99139  |

Main Findings

System C1 reports more or less the same performance as system B1. Similarly, system C2 does not report any significant difference 
from system B2. This implies that using Systems A1 and A2, which are trained to predict all the 16 classes, does not result in any significant performance gain when used as pre-trained models for systems c1 and c2, which are trained to predict a subset of all the 16 classes. However, evaluating these findings on other datasets and models covering more classes will be beneficial.



