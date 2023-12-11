import gradio as gr
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import evaluate
import nltk
from nltk.corpus import stopwords
import subprocess
import sys
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorWithPadding, DistilBertTokenizerFast
from transformers import TrainingArguments
from transformers import (
    BertModel,
    BertTokenizerFast,
    Trainer,
    EvalPrediction
)

nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")

# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 10

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="./results",
   evaluation_strategy="epoch",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
   weight_decay=WEIGHT_DECAY,
   save_total_limit=SAVE_TOTAL_LIM,
   num_train_epochs=NUM_EPOCHS,
   predict_with_generate=True,
   push_to_hub=False
)

model_id = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_id)
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# metric = evaluate.load("accuracy")

def tokenize_function(examples):
    return tokenizer(examples["stem"], padding="max_length", truncation=True)


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     metric = evaluate.load("accuracy")
#     return metric.compute(predictions=predictions, references=labels)

def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

   # rougeLSum expects newline after each sentence
   decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
   decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

   result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  
   return result


def training():
    dataset_id = "tomasmcz/word2vec_analogy"
    # dataset_id = "relbert/scientific_and_creative_analogy"
    # dataset_sub = "Quadruples_Kmiecik_random_split"
    print("GETTING DATASET")
    dataset = load_dataset(dataset_id)
    # dataset = dataset["train"]
    # tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    print(dataset)
    print(f"- The {dataset_id} dataset has {dataset['train'].num_rows} examples.")
    print(f"- Each example is a {type(dataset['train'][0])} with a {type(dataset['train'][0])} as value.")
    print(f"- Examples look like this: {dataset['train'][0]}")
    
    # for i in dataset["train"]:
    #     print(i["AB"], "to", i["CD"], "is", i["label"])
    
    dataset = dataset["train"].train_test_split(test_size=0.3)
    
    # We prefix our tasks with "answer the question"
    prefix = "Please answer this question: "

        
    def preprocess_function(examples):
        """Add prefix to the sentences, tokenize the text, and set the labels"""
        # The "inputs" are the tokenized answer:
        inputs = []
        # print(examples)
        # inputs = [prefix + doc for doc in examples["question"]]
        for doc in examples['word_a']:
            # print("THE DOC IS:", doc)
            # print("THE DOC IS:", examples[i]['AB'], examples[i]['CD'], examples[i]['label'])
            prompt = f"{prefix}{doc} is to "
            inputs.append(prompt)
        # inputs = [prefix + doc for doc in examples["question"]]
        for indx, doc in enumerate(examples["word_b"]):
            prompt = f"{doc} as "
            inputs[indx] += prompt
            
        for indx, doc in enumerate(examples["word_c"]):
            prompt = f"{doc} is to ___."
            inputs[indx] += prompt
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        
        # print(examples["label"], type(examples["label"]))
        
        # The "labels" are the tokenized outputs:
        labels = tokenizer(text_target=examples["word_d"], 
                            max_length=512,         
                            truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    
    
    # Map the preprocessing function across our dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    print("END DATALOADER")
    
    # print(train_examples)
        
    embeddings = finetune(tokenized_dataset)
    
    return 0


def finetune(dataset):
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    # model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model_id = "google/flan-t5-base"
    # model_id = "distilbert-base-uncased"
    # tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    device = torch.device('cuda:0')
    model = model.to(device)
    
    # training_args = TrainingArguments(output_dir="test_trainer")
    
    # USE THIS LINK
    # https://huggingface.co/blog/how-to-train-sentence-transformers
    
    # train_loss = losses.MegaBatchMarginLoss(model=model)
    # ds_train, ds_valid = dataset.train_test_split(test_size=0.2, seed=42)
    
    print("BEGIN FIT")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # evaluation_strategy="no"
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        )
    
    # model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)
    
    trainer.train()
    
    # model.save("flan-analogies")
    
    # model.save_to_hub("smhavens/bert-base-analogies")
    # accuracy = compute_metrics(eval, metric)
    return 0

def greet(name):
    return "Hello " + name + "!!"

def check_answer(guess:str):
    global guesses
    global answer
    guesses.append(guess)
    output = ""
    for guess in guesses:
        output += ("- " + guess + "\n")
    output = output[:-1]
    
    if guess.lower() == answer.lower():
        return "Correct!", output
    else:
        return "Try again!", output

def main():
    print("BEGIN")
    word1 = "Black"
    word2 = "White"
    word3 = "Sun"
    global answer
    answer = "Moon"
    global guesses
    
    training()
    
    


    
if __name__ == "__main__":
    main()