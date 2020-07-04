import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import nltk
nltk.download("punkt")

from nltk.tokenize import word_tokenize
from lib import *

possible_features = ["o", "n", "v", "c"]

def generateVectorFeatures(files):
    '''
    take as argument the slimmed comment data where the column 'body' is the text and return the
    the top 50 words and count how often they appears in each comments.
    '''
    porter = PorterStemmer()

    print("Cleaning Data for Vectorizer")
    all_data = []
    for file in files:
        data = pd.read_csv(file, dtype="object")
        data = html_to_char(data)
        all_data.append(data)

    all_rows = []
    sep_rows = []
    for data in all_data:
        rows = data.body.to_numpy()

        clean_rows = []
        for row in rows:
            row = row.lower()
            row = cleanText(row)
            word_tokens = word_tokenize(row)
            filtered_sentence = [porter.stem(w) for w in word_tokens if not w in stop_words]
            clean_rows.append(" ".join(filtered_sentence))

        sep_rows.append(clean_rows)
        all_rows.extend(clean_rows)



    print("Vectorizing Data")
    vectorizer = CountVectorizer(input='content', analyzer='word', lowercase=False, max_features=50)
    vectorizer.fit(all_rows)

    vectorized_rows = []
    for row in sep_rows:
        vector = vectorizer.transform(row).toarray()
        vectorized_rows.append(vector)


    return vectorized_rows


def generateCountFeatures(data):
    rows = data.body.to_numpy()

    features = []
    for text in rows:
        lexical_diversity = getLexicalDiversity(text)
        char_count = len(text)
        link_count = getNumLinks(text)
        quote_count = getNumQuotes(text)
        questions_count = getNumQuestions(text)
        bold_count = getNumBold(text)
        avgSentences_count = getNumAvgSentences(text)
        enumeration_count = getNumEnumeration(text)
        excla_count = getNumExcla(text)

        new_row = [lexical_diversity, char_count, link_count,
            quote_count, questions_count, bold_count, avgSentences_count,
            enumeration_count, excla_count]

        features.append(new_row)

    return np.array(features, dtype="float")



def generateLanFeatures(input, word_list_input, wanted_features):
    data = pd.read_csv(input, dtype="object")
    data = html_to_char(data)

    features = []
    if 'c' in wanted_features:
        print("Generating Count Features")
        count_features = generateCountFeatures(data)
        features.append(count_features)

    if 'o' in wanted_features:
        print("Generating Old Lang Features")
        old_features = generateOldLangFeatures(data)
        features.append(old_features)


    if 'n' in wanted_features:
        print("Generating New Lang Features")
        words_list = pd.read_csv(word_list_input, dtype="object")
        new_feautres = generateNewLangFeatures(data, words_list)
        features.append(new_feautres)

    return np.hstack(features)

def generateFeatures(inputs, ouputs, word_list_input, wanted_features):

    if 'c' in wanted_features or 'o' in wanted_features or 'n' in wanted_features:
        lan_features = []
        for input in inputs:
            print("Features for", input)
            lan_features.append(generateLanFeatures(input, word_list_input, wanted_features))
        lan_features = np.array(lan_features)

    if 'v' in wanted_features:
        print("Vector Features")
        vec_features = np.array(generateVectorFeatures(inputs))

        if 'c' in wanted_features or 'o' in wanted_features or 'n' in wanted_features:
            print("Combining All Features")
            features = []
            for i in range(len(lan_features)):
                features.append(np.hstack([lan_features[i], vec_features[i]]))
            features = np.array(features)

            return features
        return vec_features
    return lan_features




if __name__ == '__main__':
    delta_input = "/home/shared/CMV/SampledData/delta_sample_data.csv"
    nodelta_input = "/home/shared/CMV/SampledData/nodelta_sample_data.csv"
    word_list_input = "../data/word_list.csv"

    output_delta = "/home/shared/CMV/FeatureData/delta_sample_feature_data.csv"
    output_nodelta = "/home/shared/CMV/FeatureData/nodelta_sample_feature_data.csv"

    delta_features, nodelta_features = generateFeatures([delta_input, nodelta_input], [output_delta, output_nodelta], word_list_input, 'con')


    delta_features = pd.DataFrame(data=delta_features, columns=None)
    nodelta_features = pd.DataFrame(data=nodelta_features, columns=None)


    print("Writing Features to File")
    delta_features.to_csv(output_delta, index=False)
    nodelta_features.to_csv(output_nodelta, index=False)


''' To read the reply count:

    print("Loading reply counts...")
    reply_counts_file = open("/home/shared/CMV/reply_counts.txt", "r").read().splitlines()
    reply_counts_dct = {}
    for line in reply_counts_file:
        lst = line.split(" ")
        if lst[0] != "":
            reply_counts_dct[lst[0].split("_")[1]] = lst[1]

'''