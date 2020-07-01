import sys
sys.path.append('modules/')
from lib import *
import csv
import pickle


reader = csv.reader(open('../../word_list.csv', 'r'))
headers = next(reader, None)
dct_keys = {0:'high arousal', 1:'low arousal', 2:'medium arousal', 3:'medium dominance', 4:'low dominance', 5:'high dominance', 6:'high valence', 7:'low valence', 8:'medium valence', 9:'examples', 10:'hedges', 11:'self references'}

dct = {}
for i in headers:
    dct[i] = []
# print(dct)

for row in reader:
   for i in range(len(row)):
       if row[i] != "":
           dct[dct_keys[i]].append(row[i])

# print(dct)

def getFieldCounts(text):
    lst = [0 for i in range(len(dct_keys))]
    for i in range(len(dct.keys())):
        for word in dct[dct_keys[i]]:
            if word in text.split(" "):
                lst[i] += 1
    return(lst)

NumWords = 200


dct_keys_lst = list(dct.keys())
header = ['author', 'parend_id', 'id', 'nested_count', 'reply_count',
            'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count', 'questions_count',
            'bold_count', 'avgSentences_count', 'enumeration_count', 'excla_count'] + dct_keys_lst


reply_counts_file = open("/home/shared/CMV/reply_counts.txt", "r").read().splitlines()

reply_counts_dct = {}
for line in reply_counts_file:
    lst = line.split(" ")
    if lst[0] != "":
        reply_counts_dct[lst[0].split("_")[1]] = lst[1]

nested_counts_file = open("/home/shared/CMV/nested_counts.txt", "r").read().splitlines()

nested_counts_dct = {}
for line in nested_counts_file:
    lst = line.split(" ")[:-1]
    if (len(lst) == 2):
        nested_counts_dct[lst[0].split("_")[1]] = lst[1]


def getFeatures(text):
    body = text
    clean_word_list = cleanText(body)

    field_counts = getFieldCounts(clean_word_list)

    lexical_diversity = getLexicalDiversity(body)
    lexical_diversity_rounded = round(100 * lexical_diversity, -1)
    char_count_rounded = round(len(body), -3)
    link_count = getNumLinks(body)
    quote_count = getNumQuotes(body)
    questions_count = getNumQuestions(body)
    bold_count = getNumBold(body)
    avgSentences_count = getNumAvgSentences(body)
    enumeration_count = getNumEnumeration(body)
    excla_count = getNumExcla(body)

    features = [0, 0,
                    lexical_diversity_rounded, char_count_rounded, link_count, quote_count, questions_count,
                    bold_count, avgSentences_count, enumeration_count, excla_count] + field_counts

    return features

# features = getFeatures("Told ya")
# print(features)

filename = "rf_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))

def getPrediction(features):
    return loaded_model.predict_proba(features)


# prediction = getPrediction([features])
# print(prediction)
