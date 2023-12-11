import gradio as gr
import math
import spacy
from datasets import load_dataset
from transformers import pipeline, T5Tokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, T5ForConditionalGeneration
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
from textwrap import fill

# !pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl'])
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model_base = "results/checkpoint-17000"
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
base_prompts = ["Sun is to Moon as ", "Black is to White as ", "Atom is to Element as",
                "Athens is to Greece as ", "Cat is to Dog as ", "Robin is to Bird as",
                "Hunger is to Ambition as "]


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
    # last_checkpoint = "./results/checkpoint-22500"

    finetuned_model = T5ForConditionalGeneration.from_pretrained(model_base)
    tokenizer = T5Tokenizer.from_pretrained(model_base)
    # model = SentenceTransformer(model_base)
    gpu_available = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_available else "cpu")
    finetuned_model = finetuned_model.to(device)
    return finetuned_model, tokenizer


def cosine_scores(model, sentence):
    global word1
    global word2
    global word3
    # sentence1 = f"{word1} is to {word2} as"
    embeddings1 = model.encode(sentence, convert_to_tensor=True)

def embeddings(model, sentences, tokenizer):
    global word1
    global word2
    global word3
    global model_base
    gpu_available = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_available else "cpu")
    # device = torch.device('cuda:0')
    # embeddings = model.encode(sentences)
    question = "Please answer to this question: " + sentences
    
    inputs = tokenizer(question, return_tensors="pt")
    
    print(inputs)
    # print(inputs.device)
    print(model.device)
    print(inputs['input_ids'].device)
    print(inputs['attention_mask'].device)
    
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    inputs['input_ids'] = inputs['input_ids'].to(device)
    
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0])
    answer = answer[6:-4]
    # print(fill(answer, width=80))
    
    print("ANSWER IS", answer)

    return answer


def random_word(model, tokenizer):
    global model_base
    vocab = tokenizer.get_vocab()
    # with open(model_base + '/vocab.txt', 'r') as file:
    line = ""
    # content = file.readlines()
    length = tokenizer.vocab_size
    # print(vocab)
    while line == "":
        rand_line = random.randrange(0, length)
        # print("TRYING TO FIND", rand_line, "OUT OF", length, "WITH VOCAB OF TYPE", type(vocab))
        for word, id in vocab.items():
            if id == rand_line and word[0].isalpha() and word not in stops and word not in ROMAN_CONSTANTS:
        # if vocab[rand_line][0].isalpha() and vocab[rand_line][:-1] not in stops and vocab[rand_line][:-1] not in ROMAN_CONSTANTS:
                line = word
            elif id == rand_line:
                print(f"{word} is not alpha or is a stop word")
    # for num, aline in enumerate(file, 1997):
    #     if random.randrange(num) and aline.isalpha():
    #         continue
    #     # elif not aline.isalpha():
            
    #     line = aline
    print(line)
    return line


def generate_prompt(model, tokenizer):
    global word1
    global word2
    global word3
    global answer
    global base_prompts
    word1 = random_word(model, tokenizer)
    # word2 = random_word()
    
    word2 = embeddings(model, f"{base_prompts[random.randint(0, len(base_prompts) - 1)]}{word1} is to ___.", tokenizer)
    word3 = random_word(model, tokenizer)
    sentence = f"{word1} is to {word2} as {word3} is to ___."
    print(sentence)
    answer = embeddings(model, sentence, tokenizer)
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
    
    model, tokenizer = get_model()
    output = ""
    protected_guess = guess
    sentence = f"{word1} is to {word2} as [MASK] is to {guess}."
   
    other_word = embeddings(model, sentence, tokenizer)
    guesses.append(guess)
    
    
    
    for guess in return_guesses:
        output += ("- " + guess + "<br>")
    
    # output = output[:-1]
    prompt = f"{word1} is to {word2} as {word3} is to ___."
    # print("IS", protected_guess, "EQUAL TO", answer, ":", protected_guess.lower() == answer.lower())
    
    if protected_guess.lower() == answer.lower():
        return_guesses.append(f"{protected_guess}: {word1} is to {word2} as {word3} is to {protected_guess}.")
        output += f"<span style='color:green'>- {return_guesses[-1]}</span><br>"
        new_prompt = generate_prompt(model, tokenizer)
        return new_prompt, "Correct!", output
    else:
        return_guess = f"{protected_guess}: {word1} is to {word2} as {other_word} is to {protected_guess}."
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
    model, tokenizer = get_model() 
    generate_prompt(model, tokenizer)
    
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