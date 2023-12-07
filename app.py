import gradio as gr
import math
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


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# answer = "Pizza"
guesses = []
answer = "Pizza"


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
    dataset_id = "ag_news"
    dataset = load_dataset(dataset_id)
    # dataset = dataset["train"]
    # tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    print(f"- The {dataset_id} dataset has {dataset['train'].num_rows} examples.")
    print(f"- Each example is a {type(dataset['train'][0])} with a {type(dataset['train'][0]['text'])} as value.")
    print(f"- Examples look like this: {dataset['train'][0]}")
    
    # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    train_examples = []
    train_data = dataset['train']['text']
    # For agility we only 1/2 of our available data
    n_examples = dataset['train'].num_rows // 2
    
    
    
    for i in range(n_examples):
        example = train_data[i]
        print(example)
        train_examples.append(InputExample(texts=[example['id'], example['text']]))
        
    # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    
        
    embeddings = finetune(train_examples)
    
    return (dataset['train'].num_rows, type(dataset['train'][0]), type(dataset['train'][0]['text']), dataset['train'][0], embeddings)


def finetune(train_dataloader):
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_id)
    
    # training_args = TrainingArguments(output_dir="test_trainer")
    
    # USE THIS LINK
    # https://huggingface.co/blog/how-to-train-sentence-transformers
    
    train_loss = losses.TripletLoss(model=model)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)
    
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
    
    sentences = ["This is an example sentence", "Each sentence is converted"]

    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    print(embeddings)
    
    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

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
    
    num_rows, data_type, value, example, embeddings = training()
    
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
            gr.Markdown(f"""Number of rows in dataset is {num_rows}, with each having type {data_type} and value {value}.
                        An example is {example}.
                        The Embeddings are {embeddings}.""")
        text_button.click(check_answer, inputs=[text_input], outputs=[text_output, text_guesses])
    # iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    iface.launch()
    
    


    
if __name__ == "__main__":
    main()