---
title: AnalogyArcade
emoji: 🏆
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Model Types
### Baseline
For my dataset, I made use of  relbert/analogy_questions on huggingface, which has all data in the format of:
```
"stem": ["raphael", "painter"],
"answer": 2,
"choice": [["andersen", "plato"],
          ["reading", "berkshire"],
          ["marx", "philosopher"],
          ["tolstoi", "edison"]]
```

For a baseline, if I were to do a random selection for answer to train the system on (so the stem analogy is compared to a random choice among the answers), then there would only be a 25% baseline for correct categorization and comparison.

### Bag-of-Words Model
For comparison, I made use of my previously trained bag-of-words model from [our previous project](https://github.com/smhavens/NLPHW03).

### Fine-Tuning
#### Dataset
[analogy questions dataset](https://huggingface.co/datasets/relbert/analogy_questions)

This database uses a text with label format, with each label being an integer between 0 and 3, relating to the 4 main categories of the news:  World (0), Sports (1), Business (2), Sci/Tech (3).

I chose this one because of the larger variety of categories compared to sentiment databases, with the themes/categories theoretically being more closely related to analogies. I also chose ag_news because, as a news source, it should avoid slang and other potential hiccups that databases using tweets or general reviews will have.

#### Pre-trained model
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Because my focus is on using embeddings to evaluate analogies for the AnalogyArcade, I focused my model search for those in the sentence-transformers category, as they are readily made for embedding usage. I chose all-MiniLM-L6-v2 because of its high usage and good reviews: it is a well trained model but smaller and more efficient than its previous version.

### In-Context

## User Guide
### Introduction

### Usage

### Documentation

### Experiments

### Limitations
