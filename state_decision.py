#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

import io
import numpy
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
import math

special_chars_remover = re.compile("[^\w'|_]")
def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)

def main():
    training_sentences = read_data()
    testing_sentence = "잘도 그러겠다"
    prob_pair = naive_bayes(training_sentences, testing_sentence)
    
    print(prob_pair)

    # plot_title = testing_sentence
    # if len(plot_title) > 50: plot_title = plot_title[:50] + "..."
    
    # visualize_boxplot(plot_title,
    #               list(prob_pair),
    #               ['Negative', 'Positive'])

def naive_bayes(training_sentences, testing_sentence):
    log_prob_negative = calculate_doc_prob(training_sentences[0], testing_sentence, 0.1) + math.log(0.5)
    log_prob_positive = calculate_doc_prob(training_sentences[1], testing_sentence, 0.1) + math.log(0.5)
    prob_pair = normalize_log_prob(log_prob_negative, log_prob_positive)
    
    return prob_pair

def read_data():
    training_sentences = [[], []]
    
    '''
    여기서 파일을 읽어 training_sentences에 저장합니다.
    '''
    f = open("./ratings.txt")
    lines = f.read().split("\n")
    
    for line in lines:
        li = line.split("\t")
        if(len(li) != 3):
            continue
        if(li[2] == '0'):
            training_sentences[0].append(li[1])
        elif(li[2] == '1'):
            training_sentences[1].append(li[1])
        else:
            continue
    return [' '.join(training_sentences[0]), ' '.join(training_sentences[1])]

def normalize_log_prob(prob1, prob2):
    
    maxprob = max(prob1, prob2)

    prob1 -= maxprob
    prob2 -= maxprob
    prob1 = math.exp(prob1)
    prob2 = math.exp(prob2)

    normalize_constant = 1.0 / float(prob1 + prob2)
    prob1 *= normalize_constant
    prob2 *= normalize_constant

    return (prob1, prob2)

def calculate_doc_prob(training_sentence, testing_sentence, alpha):
    logprob = 0

    training_model = create_BOW(training_sentence)
    testing_model = create_BOW(testing_sentence)
  
    num_tokens=0
    for w in training_model:
        num_tokens += training_model[w]
    
    for word in testing_model:
        cnt = testing_model[word]
        
        if(word in training_model):
            cnt_train = training_model[word]
            logprob += cnt * (math.log(cnt_train) - math.log(num_tokens)) 
        else:
            logprob += cnt * (math.log(alpha) - math.log(num_tokens))
    
    return logprob

def create_BOW(sentence):
    bow = {}
    sentence = sentence.lower()
    sen_list = remove_special_characters(sentence).split()
    for str in sen_list:
        if len(str) < 1:
            continue
        elif str in bow.keys():
            bow[str] += 1
        else:
            bow[str] = 1

    return bow

def visualize_boxplot(title, values, labels):
    width = .35

    print(title)
    
    fig, ax = plt.subplots()
    ind = numpy.arange(len(values))
    rects = ax.bar(ind, values, width)
    ax.bar(ind, values, width=width)
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(labels)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., height + 0.01, '%.2lf%%' % (height * 100), ha='center', va='bottom')

    autolabel(rects)

    plt.savefig("image.png", format="png")

if __name__ == "__main__":
    main()