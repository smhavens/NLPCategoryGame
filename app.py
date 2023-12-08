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

# !pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl'])
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stops = stopwords.words("english")

# answer = "Pizza"
guesses = []
answer = "Pizza"


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['token_embeddings'] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops):
                lemmatized.append(lemma)
    return " ".join(lemmatized)


# def tokenize_function(examples):
#     return tokenizer(examples["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


def training():
    dataset_id = "ag_news"
    dataset = load_dataset(dataset_id)
    # dataset = dataset["train"]
    # tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    print(f"- The {dataset_id} dataset has {dataset['train'].num_rows} examples.")
    print(f"- Each example is a {type(dataset['train'][0])} with a {type(dataset['train'][0]['text'])} as value.")
    print(f"- Examples look like this: {dataset['train'][0]}")
    
    # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    # dataset = dataset["train"].map(tokenize_function, batched=True)
    # dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    # dataset.format['type']
    
    # print(dataset)
    
    train_examples = []
    train_data = dataset["train"]
    # For agility we only 1/2 of our available data
    n_examples = dataset["train"].num_rows // 2
    # n_remaining = dataset["train"].num_rows - n_examples
    # dataset_clean = {}
    # # dataset_0 = []
    # # dataset_1 = []
    # # dataset_2 = []
    # # dataset_3 = []
    # for i in range(n_examples):
    #     dataset_clean[i] = {}
    #     dataset_clean[i]["text"] = normalize(train_data[i]["text"], lowercase=True, remove_stopwords=True)
    #     dataset_clean[i]["label"] = train_data[i]["label"]
        # if train_data[i]["label"] == 0:
        #     dataset_0.append(dataset_clean[i])
        # elif train_data[i]["label"] == 1:
        #     dataset_1.append(dataset_clean[i])
        # elif train_data[i]["label"] == 2:
        #     dataset_2.append(dataset_clean[i])
        # elif train_data[i]["label"] == 3:
        #     dataset_3.append(dataset_clean[i])
    # n_0 = len(dataset_0) // 2
    # n_1 = len(dataset_1) // 2
    # n_2 = len(dataset_2) // 2
    # n_3 = len(dataset_3) // 2
    # print("Label lengths:", len(dataset_0), len(dataset_1), len(dataset_2), len(dataset_3))
    
    for i in range(n_examples):
        example = train_data[i]
        # example_opposite = dataset_clean[-(i)]
        # print(example["text"])
        train_examples.append(InputExample(texts=[example['text']], label=example['label']))
        
    # for i in range(n_0):
    #     example = dataset_0[i]
    #     # example_opposite = dataset_0[-(i)]
    #     # print(example["text"])
    #     train_examples.append(InputExample(texts=[example['text']], label=0))
        
    # for i in range(n_1):
    #     example = dataset_1[i]
    #     # example_opposite = dataset_1[-(i)]
    #     # print(example["text"])
    #     train_examples.append(InputExample(texts=[example['text']], label=1))
        
    # for i in range(n_2):
    #     example = dataset_2[i]
    #     # example_opposite = dataset_2[-(i)]
    #     # print(example["text"])
    #     train_examples.append(InputExample(texts=[example['text']], label=2))
        
    # for i in range(n_3):
    #     example = dataset_3[i]
    #     # example_opposite = dataset_3[-(i)]
    #     # print(example["text"])
    #     train_examples.append(InputExample(texts=[example['text']], label=3))
        
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=25)
    
    print("END DATALOADER")
    
    # print(train_examples)
        
    embeddings = finetune(train_dataloader)
    
    return (dataset['train'].num_rows, type(dataset['train'][0]), type(dataset['train'][0]['text']), dataset['train'][0], embeddings)


def finetune(train_dataloader):
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_id)
    
    # training_args = TrainingArguments(output_dir="test_trainer")
    
    # USE THIS LINK
    # https://huggingface.co/blog/how-to-train-sentence-transformers
    
    train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
    
    print("BEGIN FIT")
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)
    
    model.save("ag_news_model")
    
    model.save_to_hub("smhavens/all-MiniLM-agNews")
    # accuracy = compute_metrics(eval, metric)
    
    # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train,
    #     eval_dataset=eval,
    #     compute_metrics=compute_metrics,
    # )
    
    # trainer.train()
    
def embeddings():
    model = SentenceTransformer("ag_news_model")
    device = torch.device('cuda:0')
    model = model.to(device)
    sentences = ["This is an example sentence", "Each sentence is converted"]

    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    # print(embeddings)
    
    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('ag_news_model')
    # model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # print(model.device)
    # print(encoded_input["input_ids"].device)
    # print(encoded_input["attention_mask"].device)
    # print(encoded_input["token_type_ids"].device)
    encoded_input["input_ids"] = encoded_input["input_ids"].to(device)
    encoded_input["attention_mask"] = encoded_input["attention_mask"].to(device)
    encoded_input['token_type_ids'] = encoded_input['token_type_ids'].to(device)
    # print(encoded_input)
    
    # print(encoded_input["input_ids"].device)
    # print(encoded_input["attention_mask"].device)
    # print(encoded_input["token_type_ids"].device)
    
    encoded_input['input'] = {'input_ids':encoded_input['input_ids'], 'attention_mask':encoded_input['attention_mask']}
    
    #  + encoded_input['token_type_ids'] + encoded_input['attention_mask']
    del encoded_input['input_ids']
    del encoded_input['token_type_ids']
    del encoded_input['attention_mask']

    # print(encoded_input)
    
    # encoded_input.to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    print(model_output)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['input']["attention_mask"])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    print("Sentence embeddings:")
    print(sentence_embeddings)
    return sentence_embeddings

 

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
    word1 = "Black"
    word2 = "White"
    word3 = "Sun"
    global answer
    answer = "Moon"
    global guesses
    
    # num_rows, data_type, value, example, embeddings = training()
    sent_embeddings = embeddings()
    
    prompt = f"{word1} is to {word2} as {word3} is to ____"
    with gr.Blocks() as iface:
        gr.Markdown(prompt)
        with gr.Tab("Guess"):
            text_input = gr.Textbox()
            text_output = gr.Textbox()
            text_button = gr.Button("Submit")
        with gr.Accordion("Open for previous guesses"):
            text_guesses = gr.Textbox()
        with gr.Tab("Testing"):
            gr.Markdown(f"""The Embeddings are {sent_embeddings}.""")
        text_button.click(check_answer, inputs=[text_input], outputs=[text_output, text_guesses])
    # iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    iface.launch()
    
    


    
if __name__ == "__main__":
    main()