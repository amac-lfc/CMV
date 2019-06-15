import csv
from random import shuffle
import numpy as np
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Lib

#optimize the loops so you only need to run through the text once

# these are the words of certainty
wOC = [ 'absolutely', 'always', 'certain', 'certainly', 'clear', 'clearly', 'commit', 'committed', 'complete', 'completed', 
        'completely', 'every', 'exact', 'exactly', 'extremist', 'extreme', 'extremely', 'forever', 'indeed', 'inevitable',
        'inevitably', 'must', 'never', 'perfect', 'perfectly', 'perfection', 'perfected', 'positive', 'positivity', 
        'positivley', 'precise', 'precisely', 'precision', 'totally', 'truly',' undeniably', 'undeniable', 'undoubtedly', 
        'undoubted', 'unquestioned', 'unquestionably', 'unquestionable', 'unquestioning']

# this is a function that counts the number of words of certainty
getCertaintyCount = lambda text,word_i=0: text.count(wOC[word_i]) if word_i == len(wOC) - 1 else text.count(wOC[word_i]) + getCertaintyCount(text, word_i+1)

# initial words of extremity
eWC = ["much", "more", "extremely", "very", "wonderful"]

# this function counts the total of
getExtremityCount = lambda text,word_i=0: text.count(eWC[word_i]) if word_i == len(eWC) - 1 else text.count(eWC[word_i]) + getExtremityCount(text, word_i+1)

# here is the calculator for lexical diversity
def getLexicalDiversity(text):
    lstOfWords = text.split()
    setOfWords = set(lstOfWords)
    lenLstOfWords = len(lstOfWords)
    if lenLstOfWords == 0:
        return 0
    return len(setOfWords)/lenLstOfWords

# here is the counter for number of links
def getNumLinks(text):
    return text.count(r"[")

# here is the counter for num of quotes
def getNumQuotes(text):
    return text.count(">")

def getNumQuestions(text):
    return text.count("?")

def getNumBold(text):
    return text.count("**")

def getNumAvgSentences(text):
    sents = text.split('.')
    avg_len = sum(len(x.split()) for x in sents) / len(sents)
    return int(avg_len)

def cleanText(text):
    punc = '\"\\/;:,.!?\n><()[]{}-'
    for char in text:
        if (char in punc):
            text = text.replace(char, " ")
    return text

def getCommonWords(num_of_words, text):
    word_counts = {}
    text_words = text.split()
    for word in text_words:
        if word not in word_counts.keys():
            word_counts[word] = 1
        else:
            word_counts[word] += 1
    word_counts = sorted([[v,k] for k,v in word_counts.items()], reverse = True)
    result = []
    num_of_words = min(len(word_counts), num_of_words)
    for i in range(num_of_words):
        result.append(word_counts[i][1])
    return result

def simplifyCSV(input_file):
    output_file = "cleaned_" + input_file
    file = open(output_file, mode='w', encoding="utf-8")
    writer = csv.writer(file, dialect='excel', delimiter=',')

    writer.writerow(['Delta_Awarded','body'])

    with open(input_file, mode='r', encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            ", ".join(row)

            writer.writerow([row['Delta_Awarded'],row['body']])
    file.close()

def cleanCSV(input_file):
    output_file = "cleaned_" + input_file
    file = open(output_file, mode='w', encoding="utf-8")
    writer = csv.writer(file, dialect='excel', delimiter=',')

    writer.writerow(['Delta_Awarded','body'])

    with open(input_file, mode='r', encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            ", ".join(row)
            writer.writerow([row['Delta_Awarded'], cleanText(row['body'])])
    file.close()

def separateCSV(input_file):
    delta_output_file = "delta_" + input_file
    nodelta_output_file = "nodelta_" + input_file

    delta_file = open(delta_output_file, mode='w', encoding="utf-8")
    nodelta_file = open(nodelta_output_file, mode='w', encoding="utf-8")

    delta_writer = csv.writer(delta_file, dialect='excel', delimiter=',')
    nodelta_writer = csv.writer(nodelta_file, dialect='excel', delimiter=',')

    delta_writer.writerow(['Delta_Awarded','body'])
    nodelta_writer.writerow(['Delta_Awarded','body'])

    with open(input_file, mode='r', encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            ", ".join(row)
            if row['Delta_Awarded'] == '1':
                delta_writer.writerow([row['Delta_Awarded'], row['body']])
            elif row['Delta_Awarded'] == '0':
                nodelta_writer.writerow([row['Delta_Awarded'], row['body']])
    delta_file.close()
    nodelta_file.close()

def getImportances(classifier, X, features_list):
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print(f"{f + 1}. Feature {features_list[indices[f]]} ({importances[indices[f]]})")

    # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(X.shape[1]), importances[indices],
    #        color="r", yerr=std[indices], align="center")
    # plt.xticks(range(X.shape[1]), indices)
    # plt.xlim([-1, X.shape[1]])
    # plt.show()
