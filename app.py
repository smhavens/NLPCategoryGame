import gradio as gr
import math
import spacy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers import losses
from sentence_transformers import util
from transformers import pipeline
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
import random

# !pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl'])
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model_base = "bert-analogies"
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stops = stopwords.words("english")
ROMAN_CONSTANTS = (
            ( "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX" ),
            ( "", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" ),
            ( "", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM" ),
            ( "", "M", "MM", "MMM", "",   "",  "-",  "",    "",     ""   ),
            ( "", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix" ),
            ( "", "x", "xx", "xxx", "xl", "l", "lx", "lxx", "lxxx", "xc" ),
            ( "", "c", "cc", "ccc", "cd", "d", "dc", "dcc", "dccc", "cm" ),
            ( "", "m", "mm", "mmm", "",   "",  "-",  "",    "",     ""   ),
        )

# answer = "Pizza"
guesses = []
return_guesses = []
answer = "Moon"
word1 = "Black"
word2 = "White"
word3 = "Sun"


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
    
    
def get_model():
    global model_base
    model = SentenceTransformer(model_base)
    gpu_available = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_available else "cpu")
    model = model.to(device)
    return model


def cosine_scores(model, sentence):
    global word1
    global word2
    global word3
    # sentence1 = f"{word1} is to {word2} as"
    embeddings1 = model.encode(sentence, convert_to_tensor=True)

def embeddings(model, sentences):
    gpu_available = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_available else "cpu")
    # device = torch.device('cuda:0')
    embeddings = model.encode(sentences)
    global word1
    global word2
    global word3
    global model_base

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # token_ids = tokenizer.encode(sentences, return_tensors='pt')
    # blank_id = tokenizer.mask_token_id
    # blank_id_idx = torch.where(encoded_input["input_ids"] == blank_id)[1]
    
    encoded_input["input_ids"] = encoded_input["input_ids"].to(device)
    encoded_input["attention_mask"] = encoded_input["attention_mask"].to(device)
    encoded_input['token_type_ids'] = encoded_input['token_type_ids'].to(device)
    
    encoded_input['input'] = {'input_ids':encoded_input['input_ids'], 'attention_mask':encoded_input['attention_mask']}
    
    del encoded_input['input_ids']
    del encoded_input['token_type_ids']
    del encoded_input['attention_mask']

    with torch.no_grad():
        # output = model(encoded_input)
        print(encoded_input)
        model_output = model(**encoded_input)
        # output = model(encoded_input_topk)
    
    unmasker = pipeline('fill-mask', model=model_base)
    guesses = unmasker(sentences)
    print(guesses)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['input']["attention_mask"])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    potential_words = []
    for guess in guesses:
        temp_word = guess['token_str']
        if temp_word[0].isalpha() and temp_word not in stops and temp_word not in ROMAN_CONSTANTS:
            potential_words.append(guess['token_str'])
            
    rand_index = random.randint(0, len(potential_words) - 1)
    print("THE LENGTH OF POTENTIAL WORDS FOR", sentences, "IS", len(potential_words), "AND THE RANDOM INDEX CHOSEN IS", rand_index)
    chosen_word = potential_words[rand_index]

    return chosen_word


def random_word():
    global model_base
    with open(model_base + '/vocab.txt', 'r') as file:
        line = ""
        content = file.readlines()
        length = len(content)
        while line == "":
            rand_line = random.randrange(0, length)
            
            if content[rand_line][0].isalpha() and content[rand_line][:-1] not in stops and content[rand_line][:-1] not in ROMAN_CONSTANTS:
                line = content[rand_line]
            else:
                print(f"{content[rand_line]} is not alpha or is a stop word")
        # for num, aline in enumerate(file, 1997):
        #     if random.randrange(num) and aline.isalpha():
        #         continue
        #     # elif not aline.isalpha():
                
        #     line = aline
    print(line)
    return line[:-1]


def generate_prompt(model):
    global word1
    global word2
    global word3
    global answer
    word1 = random_word()
    # word2 = random_word()
    word2 = embeddings(model, f"{word1} is to [MASK].")
    word3 = random_word()
    sentence = f"{word1} is to {word2} as {word3} is to [MASK]."
    print(sentence)
    answer = embeddings(model, sentence)
    print("ANSWER IS", answer)
    return f"# {word1} is to {word2} as {word3} is to ___."
    # cosine_scores(model, sentence)


def greet(name):
    return "Hello " + name + "!!"

def check_answer(guess:str):
    global guesses
    global answer
    global return_guesses
    global word1
    global word2
    global word3
    
    model = get_model()
    output = ""
    protected_guess = guess
    sentence = f"{word1} is to {word2} as [MASK] is to {guess}."
   
    other_word = embeddings(model, sentence)
    guesses.append(guess)
    
    
    
    for guess in return_guesses:
        output += ("- " + guess + "<br>")
    
    # output = output[:-1]
    prompt = f"{word1} is to {word2} as {word3} is to ___."
    # print("IS", protected_guess, "EQUAL TO", answer, ":", protected_guess.lower() == answer.lower())
    
    if protected_guess.lower() == answer.lower():
        return_guesses.append(f"{word1} is to {word2} as {word3} is to {protected_guess}.")
        output += f"<span style='color:green'>- {return_guesses[-1]}</span><br>"
        new_prompt = generate_prompt(model)
        return new_prompt, "Correct!", output
    else:
        return_guess = f"{guess}: {word1} is to {word2} as {other_word} is to {protected_guess}."
        return_guesses.append(return_guess)
        output += ("- " + return_guess + " <br>")
        return prompt, "Try again!", output

def main():
    global word1
    global word2
    global word3
    global answer
    # answer = "Moon"
    global guesses
    
    
    # num_rows, data_type, value, example, embeddings = training()
    # sent_embeddings = embeddings()
    model = get_model() 
    generate_prompt(model)
    
    prompt = f"{word1} is to {word2} as {word3} is to ____"
    print(prompt)
    print("TESTING EMBEDDINGS")
    with gr.Blocks() as iface:
        mark_question = gr.Markdown(prompt)
        with gr.Tab("Guess"):
            text_input = gr.Textbox()
            text_output = gr.Textbox()
            text_button = gr.Button("Submit")
        with gr.Accordion("Open for previous guesses"):
            text_guesses = gr.Markdown()
        # with gr.Tab("Testing"):
        #     gr.Markdown(f"""The Embeddings are {sent_embeddings}.""")
        text_button.click(check_answer, inputs=[text_input], outputs=[mark_question, text_output, text_guesses])
    # iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    iface.launch()
    
    


    
if __name__ == "__main__":
    main()