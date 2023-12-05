import gradio as gr
import spacy
import math
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import torch.nn.functional as F
import numpy as np
import evaluate


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


def training():
    dataset = load_dataset("glue", "cola")
    dataset = dataset["train"]
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    
    
    finetune(small_train_dataset, small_eval_dataset)


def finetune(train, eval):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    
    training_args = TrainingArguments(output_dir="test_trainer")
    
    # USE THIS LINK
    # https://huggingface.co/blog/how-to-train-sentence-transformers
    
    
    # accuracy = compute_metrics(eval, metric)
    
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    sentences = ["This is an example sentence", "Each sentence is converted"]

    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    print(embeddings)
    
    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    print("Sentence embeddings:")
    print(sentence_embeddings)

 

def greet(name):
    return "Hello " + name + "!!"


def main():
    iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    iface.launch()


    
if __name__ == "__main__":
    main()