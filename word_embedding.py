<<<<<<< HEAD
from datasets import load_dataset
import shutil
import json
from collections import defaultdict
import multiprocessing
import gensim
from sklearn.metrics import classification_report
from gensim import corpora
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import fasttext
from gensim.test.utils import datapath
from wefe.datasets import load_bingliu
from wefe.metrics import RNSB
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.utils import plot_queries_results, run_queries
import pandas as pd
import gensim.downloader as api
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from wefe.metrics import WEAT
from wefe.datasets import load_weat
from wefe.utils import run_queries
from wefe.utils import plot_queries_results
import random
from scipy.special import expit
import math
import sys
import os
import argparse
import nltk
import scipy.sparse
import numpy as np
import string
import io
from sklearn.model_selection import train_test_split


'''STEPS FOR CODE:
1. Train word embeddings on Simple English Wikipedia;
2. Compare these to other pre-trained embeddings;
3. Quantify biases that exist in these word embeddings;
4. Use your word embeddings as features in a simple text classifier;
'''


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    # print("Hello", n, d)
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        # print(data)
        
    print(data)
    return data


def train_embeddings():
    '''TRAIN WORD EMBEDDINGS
    This will be making use of the dataset from wikipedia and the first step'''
    dataset = load_dataset("wikipedia", "20220301.simple")
    cores = multiprocessing.cpu_count()
    # check the first example of the training portion of the dataset :
    # print(dataset['train'][0])
    dataset_size = len(dataset)
    
    ### BUILD VOCAB ###
    # print(type(dataset["train"][0]))
    vocab = set()
    vocab_size = 0
    count = 0
    ## Generate vocab and split sentances and words?
    data = []
    for index, page in enumerate(dataset["train"]):
        document = page["text"]
        document = document.replace("\n", ". ")
        # print(document)
        for sent in document.split("."):
            # print("Sentance:", sent)
            new_sent = []
            clean_sent =[s for s in sent if s.isalnum() or s.isspace()]
            clean_sent = "".join(clean_sent)
            for word in clean_sent.split(" "):
                if len(word) > 0:
                    new_word = word.lower()
                    # print("Word:", new_word)
                    if new_word[0] not in string.punctuation:
                        new_sent.append(new_word)
            if len(new_sent) > 0:
                data.append(new_sent)
                # print("New Sent:", new_sent)
    
    
    for index, page in enumerate(dataset["train"]):
        # print(page["text"])
        # for text in page:
        #     print(text)
        text = page["text"]
        clean_text = [s for s in text if s.isalnum() or s.isspace()]
        clean_text = "".join(clean_text)
        clean_text = clean_text.replace("\n", " ")
        # text = text.replace('; ', ' ').replace(", ", " ").replace("\n", " ").replace(":", " ").replace(". ", " ").replace("! ", " ").replace("? ", " ").replace()
        
        for word in clean_text.split(" "):
            # print(word)
            if word != "\n" and word != " " and word not in vocab:
                vocab.add(word)
                vocab_size += 1
            # if index == 10:
            #     break
            # print(f"word #{index}/{count} is {word}")
        count += 1
            
    # print(f"There are {vocab_size} vocab words")
    
    embeddings_model = Word2Vec(
                     data,
                     epochs= 10,
                     window=10,
                     vector_size= 50)
    embeddings_model.save("word2vec.model")
    
    skip_model = Word2Vec(
                     data,
                     epochs= 10,
                     window=10,
                     vector_size= 50,
                     sg=1)
    skip_model.save("skip2vec.model")
    
    embeddings_model = Word2Vec.load("word2vec.model")
    skip_model = Word2Vec.load("skip2vec.model")
    
    # embeddings_model.train(dataset, total_examples=dataset_size, epochs=15)
    # print(embeddings_model['train'])
    # print(embeddings_model.wv["france"])
    return embeddings_model, skip_model


def get_data():
    dataset = load_dataset("wikipedia", "20220301.simple")
    cores = multiprocessing.cpu_count()
    # check the first example of the training portion of the dataset :
    # print(dataset['train'][0])
    dataset_size = len(dataset)
    
    ### BUILD VOCAB ###
    # print(type(dataset["train"][0]))
    vocab = set()
    vocab_size = 0
    count = 0
    ## Generate vocab and split sentances and words?
    data = []
    num_sents = 0
    for index, page in enumerate(dataset["train"]):
        document = page["text"]
        document = document.replace("\n", ". ")
        # print(document)
        for sent in document.split("."):
            num_sents += 1
            # print("Sentance:", sent)
            new_sent = []
            clean_sent =[s for s in sent if s.isalnum() or s.isspace()]
            clean_sent = "".join(clean_sent)
            for word in clean_sent.split(" "):
                if len(word) > 0:
                    new_word = word.lower()
                    # print("Word:", new_word)
                    if new_word[0] not in string.punctuation:
                        new_sent.append(new_word)
            if len(new_sent) > 0:
                data.append(new_sent)
                # print("New Sent:", new_sent)
                
    return data, num_sents


def compare_embeddings(cbow, skip, urban, fasttext):
    '''COMPARE EMBEDDINGS'''
    print("Most Similar to dog")
    print("cbow", cbow.wv.most_similar(positive=['dog'], negative=[], topn=2))
    print("skip", skip.wv.most_similar(positive=['dog'], negative=[], topn=2))
    print("urban", urban.most_similar(positive=['dog'], negative=[], topn=2))
    print("fasttext", fasttext.most_similar(positive=['dog'], negative=[], topn=2))
    
    print("\nMost Similar to Pizza - Pepperoni + Pretzel")
    print("cbow", cbow.wv.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    print("skip", skip.wv.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    print("urban", urban.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    print("fasttext", fasttext.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    
    print("\nMost Similar to witch - woman + man")
    print("cbow", cbow.wv.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    print("skip", skip.wv.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    print("urban", urban.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    print("fasttext", fasttext.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    
    print("\nMost Similar to mayor - town + country")
    print("cbow", cbow.wv.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    print("skip", skip.wv.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    print("urban", urban.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    print("fasttext", fasttext.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    
    print("\nMost Similar to death")
    print("cbow", cbow.wv.most_similar(positive=['death'], negative=[], topn=2))
    print("skip", skip.wv.most_similar(positive=['death'], negative=[], topn=2))
    print("urban", urban.most_similar(positive=['death'], negative=[], topn=2))
    print("fasttext", fasttext.most_similar(positive=['death'], negative=[], topn=2))


def quantify_bias(cbow, skip, urban, fasttext):
    '''QUANTIFY BIASES'''
    '''Using WEFE, RNSB'''
    
    RNSB_words = [
        ['christianity'],
        ['catholicism'],
        ['islam'],
        ['judaism'],
        ['hinduism'],
        ['buddhism'],
        ['mormonism'],
        ['scientology'],
        ['taoism']]
    
    weat_wordset = load_weat()
    
    models = [WordEmbeddingModel(cbow.wv, "CBOW"),
              WordEmbeddingModel(skip.wv, "skip-gram"),
              WordEmbeddingModel(urban, "urban dictionary"),
              WordEmbeddingModel(fasttext, "fasttext")]
    
    # Define the 10 Queries:
    # print(weat_wordset["science"])
    religions = ['christianity',
                 'catholicism',
                 'islam',
                 'judaism',
                 'hinduism',
                 'buddhism',
                 'mormonism',
                 'scientology',
                 'taoism',
                 'atheism']
    queries = [
        # Flowers vs Insects wrt Pleasant (5) and Unpleasant (5)
        Query([religions, weat_wordset['arts']],
            [weat_wordset['career'], weat_wordset['family']],
            ['Religion', 'Art'], ['Career', 'Family']),
        
        Query([religions, weat_wordset['weapons']],
            [weat_wordset['male_terms'], weat_wordset['female_terms']],
            ['Religion', 'Weapons'], ['Male terms', 'Female terms']),

    ]

    wefe_results = run_queries(WEAT,
                                queries,
                                models,
                                metric_params ={
                                    'preprocessors': [
                                        {},
                                        {'lowercase': True }
                                    ]
                                },
                                warn_not_found_words = True
                                ).T.round(2)
    
    print(wefe_results)
    plot_queries_results(wefe_results).show()


def text_classifier(cbow):
    '''SIMPLE TEXT CLASSIFIER'''
    '''For each document, average together all embeddings for the
    individual words in that document to get a new, d-dimensional representation
    of that document (this is essentially a “continuous bag-of-words”). Note that
    your input feature size is only d now, instead of the size of your entire vocabulary.
    Compare the results of training a model using these “CBOW” input features to
    your original (discrete) BOW model.'''
    pos_train_files = glob.glob('aclImdb/train/pos/*')
    neg_train_files = glob.glob('aclImdb/train/neg/*')
    # print(pos_train_files[:5])
    
    num_files_per_class = 1000
    # bow_train_files = cbow
    all_train_files = pos_train_files[:num_files_per_class] + neg_train_files[:num_files_per_class]
    # vectorizer = TfidfVectorizer(input="filename", stop_words="english")
    # vectors = vectorizer.fit_transform(all_train_files)
    d = len(cbow.wv["man"])
    vectors = np.empty([len(all_train_files), d])
    count = 0
    vocab = set()
    for doc in all_train_files:
        temp_array = avg_embeddings(doc, cbow, vocab)
        if len(temp_array) > 0:
            vectors[count] = temp_array
            count += 1
        else:
            vectors = np.delete(vectors, count)
    # vectors = np.array(avg_embeddings(doc, cbow) for doc in all_train_files)
    # print(vectors)
    # print(vocab)
    
    # len(vectorizer.vocabulary_)
    vectors[0].sum()
    # print("Vector at 0", vectors[0])
    
    X = vectors
    y = [1] * num_files_per_class + [0] * num_files_per_class
    len(y)
    
    x_0 = X[0]
    w = np.zeros(X.shape[1])
    # x_0_dense = x_0.todense()
    x_0.dot(w)
    
    w,b = sgd_for_lr_with_ce(X,y)
    # w
    
    # sorted_vocab = sorted([(k,v) for k,v in vectorizer.vocabulary_.items()],key=lambda x:x[1])
    sorted_vocab = sorted(vocab)
    # sorted_vocab = [a for (a,b) in sorted_vocab]
    
    sorted_words_weights = sorted([x for x in zip(sorted_vocab, w)], key=lambda x:x[1])
    sorted_words_weights[-50:]

    preds = predict_y_lr(w,b,X)
    
    preds
    
    w,b = sgd_for_lr_with_ce(X, y, num_passes=10)
    y_pred = predict_y_lr(w,b,X)
    print(classification_report(y, y_pred))

    # compute for dev set
    # pos_dev_files = glob.glob('aclImdb/test/pos/*')
    # neg_dev_files = glob.glob('aclImdb/test/neg/*')
    # num_dev_files_per_class = 100
    # all_dev_files = pos_dev_files[:num_dev_files_per_class] + neg_dev_files[:num_dev_files_per_class]
    # # use the same vectorizer from before! otherwise features won't line up
    # # don't fit it again, just use it to transform!
    # X_dev = vectorizer.transform(all_dev_files)
    # y_dev = [1]* num_dev_files_per_class + [0]* num_dev_files_per_class
    # # don't need new w and b, these are from out existing model
    # y_dev_pred = predict_y_lr(w,b,X_dev)
    # print(classification_report(y_dev, y_dev_pred))


def avg_embeddings(doc, model, vocab: set):
    words = []
    # remove out-of-vocabulary words
    with open(doc, "r") as file:
        for line in file:
            for word in line.split():
                words.append(word)
                vocab.add(word)
    words = [word for word in words if word in model.wv.index_to_key]
    if len(words) >= 1:
        return np.mean(model.wv[words], axis=0)
    else:
        return []
    


def sent_vec(sent, cbow):
    vector_size = cbow.wv.vector_size
    wv_res = np.zeros(vector_size)
    # print(wv_res)
    ctr = 1
    for w in sent:
        if w in cbow.wv:
            ctr += 1
            wv_res += cbow.wv[w]
    wv_res = wv_res/ctr
    return wv_res


def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    # doc = nlp(sentence)



    # print(doc)
    # print(type(doc))

    # Lemmatizing each token and converting each token into lowercase
    # mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # print(mytokens)

    # Removing stop words
    # mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return 0


def cbow_classifier(cbow, data, num_sentances):
    vocab_len = len(cbow.wv.index_to_key)
    
    embeddings = []
    embedding_dict = {}
    vocab = set(cbow.wv.index_to_key)
    
    # print("Data len", len(data))
    # print("Data at 0", data[0])

    X_temp = np.empty([len(data), 1])
    X_train_vect = np.array([np.array([cbow.wv[i] for i in ls if i in vocab])
                         for ls in data])
    X_test_vect = np.array([np.array([cbow.wv[i] for i in ls if i in vocab])
                         for ls in data])
    
    # words = [word for word in words if word in cbow.wv.index_to_key]
    for word in vocab:
        # embedding[word] = cbow.wv[word]
        embeddings.append(np.mean(cbow.wv[word], axis=0))
        embedding_dict[word] = np.mean(cbow.wv[word], axis=0)
    
    X = embeddings
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)
    
    # print(embeddings)
    # print(vocab_len)
    
    # X_train_vect_avg = []
    # for v in X_train_vect:
    #     if v.size:
    #         X_train_vect_avg.append(v.mean(axis=0))
    #     else:
    #         X_train_vect_avg.append(np.zeros(100, dtype=float))
            
    # X_test_vect_avg = []
    # for v in X_test_vect:
    #     if v.size:
    #         X_test_vect_avg.append(v.mean(axis=0))
    #     else:
    #         X_test_vect_avg.append(np.zeros(100, dtype=float))
            
    # # for i, v in enumerate(X_train_vect_avg):
    # #     print(len(data.iloc[i]), len(v))
    
    # x_0 = X_train_vect_avg[0]
    # num_files_per_class = 100
    # y = [1] * num_files_per_class + [0] * num_files_per_class
    # w = np.zeros(X_train_vect_avg.shape[1])
    # x_0_dense = x_0.todense()
    # x_0.dot(w)
    
    # w,b = sgd_for_lr_with_ce(X_train_vect_avg, y)
    # w
    
    # sorted_vocab = sorted([(k,v) for k,v in enumerate(embedding_dict)],key=lambda x:x[1])
    # sorted_vocab = [a for (a,b) in sorted_vocab]
    
    # sorted_words_weights = sorted([x for x in zip(sorted_vocab, w)], key=lambda x:x[1])
    # sorted_words_weights[-50:]

    # preds = predict_y_lr(w,b,X_train_vect_avg)
    
    # preds
    
    # w,b = sgd_for_lr_with_ce(X_train_vect_avg, y, num_passes=10)
    # y_pred = predict_y_lr(w,b,X_train_vect_avg)
    # print(classification_report(y, y_pred))

    # # compute for dev set
    # pos_dev_files = glob.glob('aclImdb/test/pos/*')
    # neg_dev_files = glob.glob('aclImdb/test/neg/*')
    # num_dev_files_per_class = 100
    # all_dev_files = pos_dev_files[:num_dev_files_per_class] + neg_dev_files[:num_dev_files_per_class]
    # # use the same vectorizer from before! otherwise features won't line up
    # # don't fit it again, just use it to transform!
    # # X_dev = vectorizer.transform(all_dev_files)
    # # y_dev = [1]* num_dev_files_per_class + [0]* num_dev_files_per_class
    # # # don't need new w and b, these are from out existing model
    # # y_dev_pred = predict_y_lr(w,b,X_dev)
    # # print(classification_report(y_dev, y_dev_pred))


def sgd_for_lr_with_ce(X, y, num_passes=5, learning_rate = 0.1):

    num_data_points = X.shape[0]

    # Initialize theta -> 0
    num_features = X.shape[1]
    w = np.zeros(num_features)
    b = 0.0

    # repeat until done
    # how to define "done"? let's just make it num passes for now
    # we can also do norm of gradient and when it is < epsilon (something tiny)
    # we stop

    for current_pass in range(num_passes):
        
        # iterate through entire dataset in random order
        order = list(range(num_data_points))
        random.shuffle(order)
        for i in order:

            # compute y-hat for this value of i given y_i and x_i
            x_i = X[i]
            y_i = y[i]

            # need to compute based on w and b
            # sigmoid(w dot x + b)
            z = x_i.dot(w) + b
            y_hat_i = expit(z)

            # for each w (and b), modify by -lr * (y_hat_i - y_i) * x_i
            w = w - learning_rate * (y_hat_i - y_i) * x_i
            b = b - learning_rate * (y_hat_i - y_i)

    # return theta
    return w,b


def predict_y_lr(w,b,X,threshold=0.5):

    # use our matrix operation version of the logistic regression model
    # X dot w + b
    # need to make w a column vector so the dimensions line up correctly
    y_hat = X.dot( w.reshape((-1,1)) ) + b

    # then just check if it's > threshold
    preds = np.where(y_hat > threshold,1,0)

    return preds


def main():
    parser = argparse.ArgumentParser(
        prog='word_embedding',
        description='This program will train a word embedding model using simple wikipedia.',
        epilog='To skip training the model and to used the saved model "word2vec.model", use the command --skip or -s.'
    )
    parser.add_argument('-s', '--skip', action='store_true')
    parser.add_argument('-e', '--extra', action='store_true')
    parser.add_argument('-b', '--bias', action='store_true')
    parser.add_argument('-c', '--compare', action='store_true')
    parser.add_argument('-t', '--text', action='store_true')
    
    args = parser.parse_args()
    skip_model = None
    cbow_model = None
    ud_model = None
    wiki_model = None
    if args.compare:
        if args.skip:
            # print("Skipping")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            ud_model = KeyedVectors.load("urban2vec.model")
            wiki_model = KeyedVectors.load("wiki2vec.model")
        elif args.extra:
            # print("Extra mode")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False) 
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False) 
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        else:
            cbow_model, skip_model = train_embeddings()
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False)
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False)
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        compare_embeddings(cbow_model, skip_model, ud_model, wiki_model)
    if args.bias:
        if args.skip:
            # print("Skipping")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            ud_model = KeyedVectors.load("urban2vec.model")
            wiki_model = KeyedVectors.load("wiki2vec.model")
        elif args.extra:
            # print("Extra mode")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False) 
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False) 
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        else:
            cbow_model, skip_model = train_embeddings()
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False)
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False)
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        quantify_bias(cbow_model, skip_model, ud_model, wiki_model)
    if args.text:
        if args.skip:
            # print("Skipping")
            cbow_model = Word2Vec.load("word2vec.model")
        else:
            cbow_model, skip_model = train_embeddings()
            
        text_classifier(cbow_model)
        # data, sents = get_data()
        # cbow_classifier(cbow_model, data, sents)
        
    # print("No errors?")
    

if __name__ == "__main__":
=======
from datasets import load_dataset
import shutil
import json
from collections import defaultdict
import multiprocessing
import gensim
from sklearn.metrics import classification_report
from gensim import corpora
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import fasttext
from gensim.test.utils import datapath
from wefe.datasets import load_bingliu
from wefe.metrics import RNSB
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.utils import plot_queries_results, run_queries
import pandas as pd
import gensim.downloader as api
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from wefe.metrics import WEAT
from wefe.datasets import load_weat
from wefe.utils import run_queries
from wefe.utils import plot_queries_results
import random
from scipy.special import expit
import math
import sys
import os
import argparse
import nltk
import scipy.sparse
import numpy as np
import string
import io
from sklearn.model_selection import train_test_split


'''STEPS FOR CODE:
1. Train word embeddings on Simple English Wikipedia;
2. Compare these to other pre-trained embeddings;
3. Quantify biases that exist in these word embeddings;
4. Use your word embeddings as features in a simple text classifier;
'''


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    # print("Hello", n, d)
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        # print(data)
        
    print(data)
    return data


def train_embeddings():
    '''TRAIN WORD EMBEDDINGS
    This will be making use of the dataset from wikipedia and the first step'''
    dataset = load_dataset("wikipedia", "20220301.simple")
    cores = multiprocessing.cpu_count()
    # check the first example of the training portion of the dataset :
    # print(dataset['train'][0])
    dataset_size = len(dataset)
    
    ### BUILD VOCAB ###
    # print(type(dataset["train"][0]))
    vocab = set()
    vocab_size = 0
    count = 0
    ## Generate vocab and split sentances and words?
    data = []
    for index, page in enumerate(dataset["train"]):
        document = page["text"]
        document = document.replace("\n", ". ")
        # print(document)
        for sent in document.split("."):
            # print("Sentance:", sent)
            new_sent = []
            clean_sent =[s for s in sent if s.isalnum() or s.isspace()]
            clean_sent = "".join(clean_sent)
            for word in clean_sent.split(" "):
                if len(word) > 0:
                    new_word = word.lower()
                    # print("Word:", new_word)
                    if new_word[0] not in string.punctuation:
                        new_sent.append(new_word)
            if len(new_sent) > 0:
                data.append(new_sent)
                # print("New Sent:", new_sent)
    
    
    for index, page in enumerate(dataset["train"]):
        # print(page["text"])
        # for text in page:
        #     print(text)
        text = page["text"]
        clean_text = [s for s in text if s.isalnum() or s.isspace()]
        clean_text = "".join(clean_text)
        clean_text = clean_text.replace("\n", " ")
        # text = text.replace('; ', ' ').replace(", ", " ").replace("\n", " ").replace(":", " ").replace(". ", " ").replace("! ", " ").replace("? ", " ").replace()
        
        for word in clean_text.split(" "):
            # print(word)
            if word != "\n" and word != " " and word not in vocab:
                vocab.add(word)
                vocab_size += 1
            # if index == 10:
            #     break
            # print(f"word #{index}/{count} is {word}")
        count += 1
            
    # print(f"There are {vocab_size} vocab words")
    
    embeddings_model = Word2Vec(
                     data,
                     epochs= 10,
                     window=10,
                     vector_size= 50)
    embeddings_model.save("word2vec.model")
    
    skip_model = Word2Vec(
                     data,
                     epochs= 10,
                     window=10,
                     vector_size= 50,
                     sg=1)
    skip_model.save("skip2vec.model")
    
    embeddings_model = Word2Vec.load("word2vec.model")
    skip_model = Word2Vec.load("skip2vec.model")
    
    # embeddings_model.train(dataset, total_examples=dataset_size, epochs=15)
    # print(embeddings_model['train'])
    # print(embeddings_model.wv["france"])
    return embeddings_model, skip_model


def get_data():
    dataset = load_dataset("wikipedia", "20220301.simple")
    cores = multiprocessing.cpu_count()
    # check the first example of the training portion of the dataset :
    # print(dataset['train'][0])
    dataset_size = len(dataset)
    
    ### BUILD VOCAB ###
    # print(type(dataset["train"][0]))
    vocab = set()
    vocab_size = 0
    count = 0
    ## Generate vocab and split sentances and words?
    data = []
    num_sents = 0
    for index, page in enumerate(dataset["train"]):
        document = page["text"]
        document = document.replace("\n", ". ")
        # print(document)
        for sent in document.split("."):
            num_sents += 1
            # print("Sentance:", sent)
            new_sent = []
            clean_sent =[s for s in sent if s.isalnum() or s.isspace()]
            clean_sent = "".join(clean_sent)
            for word in clean_sent.split(" "):
                if len(word) > 0:
                    new_word = word.lower()
                    # print("Word:", new_word)
                    if new_word[0] not in string.punctuation:
                        new_sent.append(new_word)
            if len(new_sent) > 0:
                data.append(new_sent)
                # print("New Sent:", new_sent)
                
    return data, num_sents


def compare_embeddings(cbow, skip, urban, fasttext):
    '''COMPARE EMBEDDINGS'''
    print("Most Similar to dog")
    print("cbow", cbow.wv.most_similar(positive=['dog'], negative=[], topn=2))
    print("skip", skip.wv.most_similar(positive=['dog'], negative=[], topn=2))
    print("urban", urban.most_similar(positive=['dog'], negative=[], topn=2))
    print("fasttext", fasttext.most_similar(positive=['dog'], negative=[], topn=2))
    
    print("\nMost Similar to Pizza - Pepperoni + Pretzel")
    print("cbow", cbow.wv.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    print("skip", skip.wv.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    print("urban", urban.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    print("fasttext", fasttext.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    
    print("\nMost Similar to witch - woman + man")
    print("cbow", cbow.wv.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    print("skip", skip.wv.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    print("urban", urban.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    print("fasttext", fasttext.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    
    print("\nMost Similar to mayor - town + country")
    print("cbow", cbow.wv.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    print("skip", skip.wv.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    print("urban", urban.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    print("fasttext", fasttext.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    
    print("\nMost Similar to death")
    print("cbow", cbow.wv.most_similar(positive=['death'], negative=[], topn=2))
    print("skip", skip.wv.most_similar(positive=['death'], negative=[], topn=2))
    print("urban", urban.most_similar(positive=['death'], negative=[], topn=2))
    print("fasttext", fasttext.most_similar(positive=['death'], negative=[], topn=2))


def quantify_bias(cbow, skip, urban, fasttext):
    '''QUANTIFY BIASES'''
    '''Using WEFE, RNSB'''
    
    RNSB_words = [
        ['christianity'],
        ['catholicism'],
        ['islam'],
        ['judaism'],
        ['hinduism'],
        ['buddhism'],
        ['mormonism'],
        ['scientology'],
        ['taoism']]
    
    weat_wordset = load_weat()
    
    models = [WordEmbeddingModel(cbow.wv, "CBOW"),
              WordEmbeddingModel(skip.wv, "skip-gram"),
              WordEmbeddingModel(urban, "urban dictionary"),
              WordEmbeddingModel(fasttext, "fasttext")]
    
    # Define the 10 Queries:
    # print(weat_wordset["science"])
    religions = ['christianity',
                 'catholicism',
                 'islam',
                 'judaism',
                 'hinduism',
                 'buddhism',
                 'mormonism',
                 'scientology',
                 'taoism',
                 'atheism']
    queries = [
        # Flowers vs Insects wrt Pleasant (5) and Unpleasant (5)
        Query([religions, weat_wordset['arts']],
            [weat_wordset['career'], weat_wordset['family']],
            ['Religion', 'Art'], ['Career', 'Family']),
        
        Query([religions, weat_wordset['weapons']],
            [weat_wordset['male_terms'], weat_wordset['female_terms']],
            ['Religion', 'Weapons'], ['Male terms', 'Female terms']),

    ]

    wefe_results = run_queries(WEAT,
                                queries,
                                models,
                                metric_params ={
                                    'preprocessors': [
                                        {},
                                        {'lowercase': True }
                                    ]
                                },
                                warn_not_found_words = True
                                ).T.round(2)
    
    print(wefe_results)
    plot_queries_results(wefe_results).show()


def text_classifier(cbow):
    '''SIMPLE TEXT CLASSIFIER'''
    '''For each document, average together all embeddings for the
    individual words in that document to get a new, d-dimensional representation
    of that document (this is essentially a “continuous bag-of-words”). Note that
    your input feature size is only d now, instead of the size of your entire vocabulary.
    Compare the results of training a model using these “CBOW” input features to
    your original (discrete) BOW model.'''
    pos_train_files = glob.glob('aclImdb/train/pos/*')
    neg_train_files = glob.glob('aclImdb/train/neg/*')
    # print(pos_train_files[:5])
    
    num_files_per_class = 1000
    # bow_train_files = cbow
    all_train_files = pos_train_files[:num_files_per_class] + neg_train_files[:num_files_per_class]
    # vectorizer = TfidfVectorizer(input="filename", stop_words="english")
    # vectors = vectorizer.fit_transform(all_train_files)
    d = len(cbow.wv["man"])
    vectors = np.empty([len(all_train_files), d])
    count = 0
    vocab = set()
    for doc in all_train_files:
        temp_array = avg_embeddings(doc, cbow, vocab)
        if len(temp_array) > 0:
            vectors[count] = temp_array
            count += 1
        else:
            vectors = np.delete(vectors, count)
    # vectors = np.array(avg_embeddings(doc, cbow) for doc in all_train_files)
    # print(vectors)
    # print(vocab)
    
    # len(vectorizer.vocabulary_)
    vectors[0].sum()
    # print("Vector at 0", vectors[0])
    
    X = vectors
    y = [1] * num_files_per_class + [0] * num_files_per_class
    len(y)
    
    x_0 = X[0]
    w = np.zeros(X.shape[1])
    # x_0_dense = x_0.todense()
    x_0.dot(w)
    
    w,b = sgd_for_lr_with_ce(X,y)
    # w
    
    # sorted_vocab = sorted([(k,v) for k,v in vectorizer.vocabulary_.items()],key=lambda x:x[1])
    sorted_vocab = sorted(vocab)
    # sorted_vocab = [a for (a,b) in sorted_vocab]
    
    sorted_words_weights = sorted([x for x in zip(sorted_vocab, w)], key=lambda x:x[1])
    sorted_words_weights[-50:]

    preds = predict_y_lr(w,b,X)
    
    preds
    
    w,b = sgd_for_lr_with_ce(X, y, num_passes=10)
    y_pred = predict_y_lr(w,b,X)
    print(classification_report(y, y_pred))

    # compute for dev set
    # pos_dev_files = glob.glob('aclImdb/test/pos/*')
    # neg_dev_files = glob.glob('aclImdb/test/neg/*')
    # num_dev_files_per_class = 100
    # all_dev_files = pos_dev_files[:num_dev_files_per_class] + neg_dev_files[:num_dev_files_per_class]
    # # use the same vectorizer from before! otherwise features won't line up
    # # don't fit it again, just use it to transform!
    # X_dev = vectorizer.transform(all_dev_files)
    # y_dev = [1]* num_dev_files_per_class + [0]* num_dev_files_per_class
    # # don't need new w and b, these are from out existing model
    # y_dev_pred = predict_y_lr(w,b,X_dev)
    # print(classification_report(y_dev, y_dev_pred))


def avg_embeddings(doc, model, vocab: set):
    words = []
    # remove out-of-vocabulary words
    with open(doc, "r") as file:
        for line in file:
            for word in line.split():
                words.append(word)
                vocab.add(word)
    words = [word for word in words if word in model.wv.index_to_key]
    if len(words) >= 1:
        return np.mean(model.wv[words], axis=0)
    else:
        return []
    


def sent_vec(sent, cbow):
    vector_size = cbow.wv.vector_size
    wv_res = np.zeros(vector_size)
    # print(wv_res)
    ctr = 1
    for w in sent:
        if w in cbow.wv:
            ctr += 1
            wv_res += cbow.wv[w]
    wv_res = wv_res/ctr
    return wv_res


def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    # doc = nlp(sentence)



    # print(doc)
    # print(type(doc))

    # Lemmatizing each token and converting each token into lowercase
    # mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # print(mytokens)

    # Removing stop words
    # mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return 0


def cbow_classifier(cbow, data, num_sentances):
    vocab_len = len(cbow.wv.index_to_key)
    
    embeddings = []
    embedding_dict = {}
    vocab = set(cbow.wv.index_to_key)
    
    # print("Data len", len(data))
    # print("Data at 0", data[0])

    X_temp = np.empty([len(data), 1])
    X_train_vect = np.array([np.array([cbow.wv[i] for i in ls if i in vocab])
                         for ls in data])
    X_test_vect = np.array([np.array([cbow.wv[i] for i in ls if i in vocab])
                         for ls in data])
    
    # words = [word for word in words if word in cbow.wv.index_to_key]
    for word in vocab:
        # embedding[word] = cbow.wv[word]
        embeddings.append(np.mean(cbow.wv[word], axis=0))
        embedding_dict[word] = np.mean(cbow.wv[word], axis=0)
    
    X = embeddings
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)
    
    # print(embeddings)
    # print(vocab_len)
    
    # X_train_vect_avg = []
    # for v in X_train_vect:
    #     if v.size:
    #         X_train_vect_avg.append(v.mean(axis=0))
    #     else:
    #         X_train_vect_avg.append(np.zeros(100, dtype=float))
            
    # X_test_vect_avg = []
    # for v in X_test_vect:
    #     if v.size:
    #         X_test_vect_avg.append(v.mean(axis=0))
    #     else:
    #         X_test_vect_avg.append(np.zeros(100, dtype=float))
            
    # # for i, v in enumerate(X_train_vect_avg):
    # #     print(len(data.iloc[i]), len(v))
    
    # x_0 = X_train_vect_avg[0]
    # num_files_per_class = 100
    # y = [1] * num_files_per_class + [0] * num_files_per_class
    # w = np.zeros(X_train_vect_avg.shape[1])
    # x_0_dense = x_0.todense()
    # x_0.dot(w)
    
    # w,b = sgd_for_lr_with_ce(X_train_vect_avg, y)
    # w
    
    # sorted_vocab = sorted([(k,v) for k,v in enumerate(embedding_dict)],key=lambda x:x[1])
    # sorted_vocab = [a for (a,b) in sorted_vocab]
    
    # sorted_words_weights = sorted([x for x in zip(sorted_vocab, w)], key=lambda x:x[1])
    # sorted_words_weights[-50:]

    # preds = predict_y_lr(w,b,X_train_vect_avg)
    
    # preds
    
    # w,b = sgd_for_lr_with_ce(X_train_vect_avg, y, num_passes=10)
    # y_pred = predict_y_lr(w,b,X_train_vect_avg)
    # print(classification_report(y, y_pred))

    # # compute for dev set
    # pos_dev_files = glob.glob('aclImdb/test/pos/*')
    # neg_dev_files = glob.glob('aclImdb/test/neg/*')
    # num_dev_files_per_class = 100
    # all_dev_files = pos_dev_files[:num_dev_files_per_class] + neg_dev_files[:num_dev_files_per_class]
    # # use the same vectorizer from before! otherwise features won't line up
    # # don't fit it again, just use it to transform!
    # # X_dev = vectorizer.transform(all_dev_files)
    # # y_dev = [1]* num_dev_files_per_class + [0]* num_dev_files_per_class
    # # # don't need new w and b, these are from out existing model
    # # y_dev_pred = predict_y_lr(w,b,X_dev)
    # # print(classification_report(y_dev, y_dev_pred))


def sgd_for_lr_with_ce(X, y, num_passes=5, learning_rate = 0.1):

    num_data_points = X.shape[0]

    # Initialize theta -> 0
    num_features = X.shape[1]
    w = np.zeros(num_features)
    b = 0.0

    # repeat until done
    # how to define "done"? let's just make it num passes for now
    # we can also do norm of gradient and when it is < epsilon (something tiny)
    # we stop

    for current_pass in range(num_passes):
        
        # iterate through entire dataset in random order
        order = list(range(num_data_points))
        random.shuffle(order)
        for i in order:

            # compute y-hat for this value of i given y_i and x_i
            x_i = X[i]
            y_i = y[i]

            # need to compute based on w and b
            # sigmoid(w dot x + b)
            z = x_i.dot(w) + b
            y_hat_i = expit(z)

            # for each w (and b), modify by -lr * (y_hat_i - y_i) * x_i
            w = w - learning_rate * (y_hat_i - y_i) * x_i
            b = b - learning_rate * (y_hat_i - y_i)

    # return theta
    return w,b


def predict_y_lr(w,b,X,threshold=0.5):

    # use our matrix operation version of the logistic regression model
    # X dot w + b
    # need to make w a column vector so the dimensions line up correctly
    y_hat = X.dot( w.reshape((-1,1)) ) + b

    # then just check if it's > threshold
    preds = np.where(y_hat > threshold,1,0)

    return preds


def main():
    parser = argparse.ArgumentParser(
        prog='word_embedding',
        description='This program will train a word embedding model using simple wikipedia.',
        epilog='To skip training the model and to used the saved model "word2vec.model", use the command --skip or -s.'
    )
    parser.add_argument('-s', '--skip', action='store_true')
    parser.add_argument('-e', '--extra', action='store_true')
    parser.add_argument('-b', '--bias', action='store_true')
    parser.add_argument('-c', '--compare', action='store_true')
    parser.add_argument('-t', '--text', action='store_true')
    
    args = parser.parse_args()
    skip_model = None
    cbow_model = None
    ud_model = None
    wiki_model = None
    if args.compare:
        if args.skip:
            # print("Skipping")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            ud_model = KeyedVectors.load("urban2vec.model")
            wiki_model = KeyedVectors.load("wiki2vec.model")
        elif args.extra:
            # print("Extra mode")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False) 
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False) 
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        else:
            cbow_model, skip_model = train_embeddings()
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False)
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False)
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        compare_embeddings(cbow_model, skip_model, ud_model, wiki_model)
    if args.bias:
        if args.skip:
            # print("Skipping")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            ud_model = KeyedVectors.load("urban2vec.model")
            wiki_model = KeyedVectors.load("wiki2vec.model")
        elif args.extra:
            # print("Extra mode")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False) 
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False) 
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        else:
            cbow_model, skip_model = train_embeddings()
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False)
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False)
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        quantify_bias(cbow_model, skip_model, ud_model, wiki_model)
    if args.text:
        if args.skip:
            # print("Skipping")
            cbow_model = Word2Vec.load("word2vec.model")
        else:
            cbow_model, skip_model = train_embeddings()
            
        text_classifier(cbow_model)
        # data, sents = get_data()
        # cbow_classifier(cbow_model, data, sents)
        
    # print("No errors?")
    

if __name__ == "__main__":
>>>>>>> 7d5b505 (New in-context model with working UI System)
    main()