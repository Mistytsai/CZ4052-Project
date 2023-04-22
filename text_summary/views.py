import json

from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import pandas as pd
from rouge import Rouge
import matplotlib.pyplot as plt

DAMPING_FACTOR = 0.85
MAX_ITER = 100
EPSILON = 1e-5


def cal_similarity(str1, str2):
    co_occur_count = len(set(str1) & set(str2))
    if (abs(co_occur_count) <= 1e-12) or (len(str1) == 0 or len(str1) == 0):
        return 0.
    denominator = math.log(float(len(str1))) + math.log(float(len(str2)))
    if abs(denominator) < 1e-12:
        return 0.
    return co_occur_count / denominator


def get_adj_matrix(sentence_tokens):
    tokens_count = len(sentence_tokens)
    adj_matrix = np.zeros((tokens_count, tokens_count))
    for i in range(tokens_count):
        for j in range(i, tokens_count):
            s = cal_similarity(sentence_tokens[i], sentence_tokens[j])
            adj_matrix[i, j] = s
            adj_matrix[j, i] = s
    return adj_matrix




def cal_score(adj_matrix, d=DAMPING_FACTOR, max_iter=MAX_ITER, threshold=EPSILON):
    N = len(adj_matrix)
    # Sum all the values of each column
    col_sum = adj_matrix.sum(axis=0).astype(float)
    # Replace any zero value in col_sum with 0.001 to prevent "division by zero" error.
    col_sum[col_sum == 0.0] = 0.001
    # Normalize the adjacency matrix (divide by sum of each column)
    adj_matrix = adj_matrix / col_sum

    # Initialize PageRank vector, each with a value of 1/N
    pr = np.full([N, 1], 1 / N)
    for _ in range(max_iter):
        prev_pr = pr.copy()
        pr = (1 - d) + d * np.dot(adj_matrix.T, pr)
        # Check for convergence by computing the L1 norm of the difference between the previous and current PageRank vectors
        diff = np.abs(pr - prev_pr).sum()
        if diff < threshold:
            break
    # Normalization, to ensure sum of all scores is one.
    pr = pr / pr.sum()
    # Create a dictionary that maps each node index to its corresponding PageRank score.
    scores = dict(zip(range(len(pr)), [i[0] for i in pr]))
    return scores

def get_sorted_sentences(scores, index_items):
    items_scores = dict()
    for index, score in scores.items():
        items_scores[index_items.get(index)] = score
    sorted_items = sorted(items_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_items


def get_top_n_indices(scores, n):
    # Convert the scores dictionary into a list of tuples (sentence index, score)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Get the indices of the top n sentences
    top_indices = [index for index, score in sorted_scores[:n]]
    # Sort the indices in their original order
    top_indices.sort()
    return top_indices



def pre_process(sentences):
    stop_words = set(stopwords.words('english'))
    clean_sentences = []
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        clean_sentences.append(' '.join(words))
    return clean_sentences


def textrank(text, reduce_rate=0.7, d=DAMPING_FACTOR, max_iter=MAX_ITER, threshold=EPSILON):

    sentences = sent_tokenize(text)
    sentence_tokens = pre_process(sentences)
    adj_matrix = get_adj_matrix(sentence_tokens)
    scores = cal_score(adj_matrix, d, max_iter, threshold)
    n = math.ceil(len(sentence_tokens) * (1 - reduce_rate))
    indices = get_top_n_indices(scores, n)

    summary = ' '.join([sentences[i] for i in indices])
    return summary



def index(request):
    if request.method=='GET':
        return render(request,'index.html')
    else:
        data = json.loads(request.body)
        size = data.get('size')
        text = data.get('text')
        reduce_rate = 0.7
        min_sentences = 3
        sentence_count = len(sent_tokenize(text))
        if sentence_count <= min_sentences:
            reduce_rate = 0
        if size == 'small':
             reduce_rate = max(0.3, min(0.8, 1 - (min_sentences / sentence_count)))
        elif size == 'verbose':
            reduce_rate = min(0.4, 1 - (min_sentences / sentence_count) * 2)
        summary = textrank(text, reduce_rate)
        return HttpResponse({summary})


