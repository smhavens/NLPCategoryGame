import gradio as gr
import math
import spacy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers import losses
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import evaluate
import nltk
from nltk.corpus import stopwords
import subprocess
import sys
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import (
    BertModel,
    BertTokenizerFast,
    Trainer,
    EvalPrediction
)

# !pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl'])
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# nltk.download('stopwords')
# nlp = spacy.load("en_core_web_sm")
# stops = stopwords.words("english")

# answer = "Pizza"
guesses = []
answer = "Pizza"

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
metric = evaluate.load("accuracy")

def tokenize_function(examples):
    return tokenizer(examples["stem"], padding="max_length", truncation=True)


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


# def training():
#     dataset_id = "relbert/analogy_questions"
#     dataset_sub = "bats"
#     print("GETTING DATASET")
#     raw_dataset = load_dataset(dataset_id, dataset_sub)
#     # data_metric = evaluate.load(dataset_id, dataset_sub)
#     checkpoint = "bert-base-uncased"
#     model = BertModel.from_pretrained(checkpoint)
#     # dataset = dataset["train"]
#     # tokenized_datasets = dataset.map(tokenize_function, batched=True)
#     # print(raw_dataset)
#     test_data = raw_dataset["test"]
#     # print(test_data["stem"])
#     all_answers = []
#     for answer in raw_dataset["answer"]:
#         answer = raw_dataset["choice"][answer]
#     raw_dataset = raw_dataset.add_column("label", all_answers)
    
        
#     print(raw_dataset)
#     print(raw_dataset["label"])
#     dataset = raw_dataset.map(
#         lambda x: tokenizer(x["stem"], truncation=True),
#         batched=True,
#     )
#     print(dataset)
#     dataset = dataset.remove_columns(["stem", "answer", "choice"])
#     dataset = dataset.rename_column("label", "labels")
#     dataset = dataset.with_format("torch")

#     training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

#     print(dataset)
#     # print(f"- The {dataset_id} dataset has {dataset.num_rows} examples.")
#     # print(f"- Each example is a {type(dataset[0])} with a {type(dataset[0]['stem'])} as value.")
#     # print(f"- Examples look like this: {dataset[0]}")
    
#     # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
#     # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
#     # dataset = dataset["train"].map(tokenize_function, batched=True)
#     # dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
#     # dataset.format['type']
    
#     # tokenized_news = dataset.map(tokenize_function, batched=True)
    
#     # model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", num_labels=2)
    
#     # print(dataset)
    
#     # Choose the appropriate device based on availability (CUDA or CPU)
#     # gpu_available = torch.cuda.is_available()
#     # device = torch.device("cuda" if gpu_available else "cpu")
#     # model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    
#     # tokenized_datasets = dataset.map(tokenize_function, batched=True)
#     # print(tokenized_datasets)
#     # # small_train_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
#     # # small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))
    
#     # model = model.to(device)
    
#     # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
#     # training_args = TrainingArguments(output_dir="test_trainer")
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset["test"],
#         eval_dataset=dataset["validation"],
#         compute_metrics=compute_metrics,
#     )
    
#     output = trainer.train()
    
#     # train_examples = []
#     # train_data = dataset["train"]
#     # # For agility we only 1/2 of our available data
#     # n_examples = dataset["train"].num_rows // 2
    
#     # for i in range(n_examples):
#     #     example = train_data[i]
#     #     # example_opposite = dataset_clean[-(i)]
#     #     # print(example["text"])
#     #     train_examples.append(InputExample(texts=[example['stem'], example]))
    
        
#     # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=25)
    
#     # print("END DATALOADER")
    
#     # # print(train_examples)
        
#     # embeddings = finetune(train_dataloader)
#     print(output)
    
#     model.save("bert-analogies")
    
#     model.save_to_hub("smhavens/bert-base-analogies")
#     return output


# def finetune(train_dataloader):
#     # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
#     model_id = "sentence-transformers/all-MiniLM-L6-v2"
#     model = SentenceTransformer(model_id)
#     device = torch.device('cuda:0')
#     model = model.to(device)
    
#     # training_args = TrainingArguments(output_dir="test_trainer")
    
#     # USE THIS LINK
#     # https://huggingface.co/blog/how-to-train-sentence-transformers
    
#     train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
    
#     print("BEGIN FIT")
    
#     model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)
    
#     model.save("bert-analogies")
    
#     model.save_to_hub("smhavens/bert-base-analogies")
#     return 0

def training():
    dataset_id = "relbert/analogy_questions"
    dataset_sub = "bats"
    print("GETTING DATASET")
    dataset = load_dataset(dataset_id, dataset_sub)
    # dataset = dataset["train"]
    # tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    print(f"- The {dataset_id} dataset has {dataset['test'].num_rows} examples.")
    print(f"- Each example is a {type(dataset['test'][0])} with a {type(dataset['test'][0]['stem'])} as value.")
    print(f"- Examples look like this: {dataset['test'][0]}")
    
    train_examples = []
    train_data = dataset["test"]
    # For agility we only 1/2 of our available data
    n_examples = dataset["test"].num_rows // 2
    
    for i in range(n_examples):
        example = train_data[i]
        temp_word_1 = example["stem"][0]
        temp_word_2 = example["stem"][1]
        temp_word_3 = example["choice"][example["answer"]][0]
        temp_word_4 = example["choice"][example["answer"]][1]
        comp1 = f"{temp_word_1} to {temp_word_2}"
        comp2 = f"{temp_word_3} to {temp_word_4}"
        # example_opposite = dataset_clean[-(i)]
        # print(example["text"])
        train_examples.append(InputExample(texts=[comp1, comp2]))
    
        
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=25)
    
    print("END DATALOADER")
    
    # print(train_examples)
        
    embeddings = finetune(train_dataloader)
    
    return (dataset['test'].num_rows, type(dataset['test'][0]), type(dataset['test'][0]['stem']), dataset['test'][0], embeddings)


def finetune(train_dataloader):
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_id)
    device = torch.device('cuda:0')
    model = model.to(device)
    
    # training_args = TrainingArguments(output_dir="test_trainer")
    
    # USE THIS LINK
    # https://huggingface.co/blog/how-to-train-sentence-transformers
    
    train_loss = losses.MegaBatchMarginLoss(model=model)
    
    print("BEGIN FIT")
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)
    
    model.save("bert-analogies")
    
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
    
    num_rows, data_type, value, example, embeddings = training()
    
    # prompt = f"{word1} is to {word2} as {word3} is to ____"
    # with gr.Blocks() as iface:
    #     gr.Markdown(prompt)
    #     with gr.Tab("Guess"):
    #         text_input = gr.Textbox()
    #         text_output = gr.Textbox()
    #         text_button = gr.Button("Submit")
    #     with gr.Accordion("Open for previous guesses"):
    #         text_guesses = gr.Textbox()
    #     with gr.Tab("Testing"):
    #         gr.Markdown(f"""Number of rows in dataset is {num_rows}, with each having type {data_type} and value {value}.
    #                     An example is {example}.
    #                     The Embeddings are {embeddings}.""")
    #     text_button.click(check_answer, inputs=[text_input], outputs=[text_output, text_guesses])
    # # iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    # iface.launch()
    
    


    
if __name__ == "__main__":
    main()