# Welcome to Learning how to Active Learn

 ## Introduction

 This source code is the basis of the following paper:
 > [Learning how to Active Learn: A Deep Reinforcement Learning Approach](http://people.eng.unimelb.edu.au/tcohn/papers/emnlp17pal.pdf), by Meng Fang, Yuan Li and Trevor Cohn, EMNLP 2017

 ## Building

 It's developed on TensorFlow.
 - Install [TensorFlow](https://www.tensorflow.org/)
 - Install [pycrfsuite](https://python-crfsuite.readthedocs.io/en/latest/index.html)

 ## Code

 - launcher_ner_bilingual: the starter of playing
 - game_ner: the game
 - robot: active learning policy
 - tagger

## How to run

For example, we train the active learning policy on English and then apply the policy to German.

```sh
python launcher_ner_bilingual.py --agent "CNNDQN" --episode 10000 --budget 1000 --train "en.train;en.testa;en.testb;en.emb;en.model.saved" --test "de.train;de.testa;de.testb;de.emb;de.model.saved"
```

## Data resource

- [CoNLL2002](http://www.cnts.ua.ac.be/conll2002/ner/) 
- [CoNLL2003](http://www.cnts.ua.ac.be/conll2003/ner/)
- [Crosslingual word embeddings and corpora](http://128.2.220.95/multilingual/data)