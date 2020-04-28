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


replacements = np.array([["&nbsp;", " "], ["&iexcl;", "¡"], ["&cent;", "¢"], ["&pound;", "£"], ["&curren;", "¤"], ["&yen;", "¥"], ["&brvbar;", "¦"], ["&sect;", "§"], ["&uml;", "¨"], ["&copy;", "©"], ["&ordf;", "ª"], ["&laquo;", "«"], ["&not;", "¬"], ["&reg;", "®"], ["&macr;", "¯"], ["&deg;", "°"], ["&plusmn;", "±"], ["&sup2;", "²"], ["&sup3;", "³"], ["&acute;", "´"], ["&micro;", "µ"], ["&para;", "¶"], ["&cedil;", "¸"], ["&sup1;", "¹"], ["&ordm;", "º"], ["&raquo;", "»"], ["&frac14;", "¼"], ["&frac12;", "½"], ["&frac34;", "3/4"], ["&iquest;", "¿"], ["&times;", "times"], ["&divide;", "divided by"], ["&Agrave;", "À"], ["&Aacute;", "Á"], ["&Acirc;", "Â"], ["&Atilde;", "Ã"], ["&Auml;", "Ä"], ["&Aring;", "Å"], ["&AElig;", "Æ"], ["&Ccedil;", "Ç"], ["&Egrave;", "È"], ["&Eacute;", "É"], ["&Ecirc;", "Ê"], ["&Euml;", "Ë"], ["&Igrave;", "Ì"], ["&Iacute;", "Í"], ["&Icirc;", "Î"], ["&Iuml;", "Ï"], ["&ETH;", "Ð"], ["&Ntilde;", "Ñ"], ["&Ograve;", "Ò"], ["&Oacute;", "Ó"], ["&Ocirc;", "Ô"], ["&Otilde;", "Õ"], ["&Ouml;", "Ö"], ["&Oslash;", "Ø"], ["&Ugrave;", "Ù"], ["&Uacute;", "Ú"], ["&Ucirc;", "Û"], ["&Uuml;", "Ü"], ["&Yacute;", "Ý"], ["&THORN;", "Þ"], ["&szlig;", "ß"], ["&agrave;", "à"], ["&aacute;", "á"], ["&acirc;", "â"], ["&atilde;", "ã"], ["&auml;", "ä"], ["&aring;", "å"], ["&aelig;", "æ"], ["&ccedil;", "ç"], ["&egrave;", "è"], ["&eacute;", "é"], ["&ecirc;", "ê"], ["&euml;", "ë"], ["&igrave;", "ì"], ["&iacute;", "í"], ["&icirc;", "î"], ["&iuml;", "ï"], ["&eth;", "ð"], ["&ntilde;", "ñ"], ["&ograve;", "ò"], ["&oacute;", "ó"], ["&ocirc;", "ô"], ["&otilde;", "õ"], ["&ouml;", "ö"], ["&oslash;", "ø"], ["&ugrave;", "ù"], ["&uacute;", "ú"], ["&ucirc;", "û"], ["&uuml;", "ü"], ["&yacute;", "ý"], ["&thorn;", "þ"], ["&yuml;", "ÿ"], ["&amp;", "&"], ["&lt;", "<"], ["&gt;", ">"], ["&OElig;", "Œ"], ["&oelig;", "œ"], ["&Scaron;", "Š"], ["&scaron;", "š"], ["&Yuml;", "Ÿ"], ["&fnof;", "ƒ"], ["&circ;", "ˆ"], ["&tilde;", "˜"], ["&ndash;", "–"], ["&mdash;", "—"], ["&lsquo;", "‘"], ["&rsquo;", "’"], ["&sbquo;", "‚"], ["&ldquo;", "“"], ["&rdquo;", "”"], ["&bdquo;", "„"], ["&dagger;", "†"], ["&Dagger;", "‡"], ["&bull;", "•"], ["&hellip;", "…"], ["&permil;", "‰"], ["&lsaquo;", "‹"], ["&rsaquo;", "›"], ["&euro;", "€"], ["&trade;", "™"]])


possible_features = ["o", "n", "v", "c"]

def html_to_char(data):
    temp = data.copy()

    for punc in replacements:
        temp['body'] = temp['body'].str.replace(punc[0], punc[1])

    return temp


def generateOldLangFeatures(data):
    rows = data.body.to_numpy()
    features = []
    for text in rows:
        certainty_count = getCertaintyCount(text)
        extremity_count = getExtremityCount(text)

        new_row = [certainty_count, extremity_count]

        features.append(new_row)

    return np.array(features, dtype="float")

def countOccurences(text, lst):
    text = text.lower()

    count = 0
    for word in lst:
        count += text.split(" ").count(word)
    return count

def dummy(value):
    return value

def generateNewLangFeatures(data, words_df):
    rows = data.body.to_numpy()
    # print(len(rows))

    count = 0
    new_rows = []
    for text in rows:
        temp_row = []
        for word_type in words_df.columns:
            word_type_lst = words_df[word_type].dropna().to_numpy()
            temp_count = countOccurences(text, word_type_lst)
            temp_row.append(temp_count)
        new_rows.append(temp_row)
        count += 1
        # print(count)

    return np.array(new_rows)


def generateVectorFeatures(files):
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
    vectorizer = CountVectorizer(input='content', analyzer='word', lowercase=False)
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
        lexical_diversity_rounded = round(100 * lexical_diversity, -1)
        char_count_rounded = round(len(text), -3)
        link_count = getNumLinks(text)
        quote_count = getNumQuotes(text)
        questions_count = getNumQuestions(text)
        bold_count = getNumBold(text)
        avgSentences_count = getNumAvgSentences(text)
        enumeration_count = getNumEnumeration(text)
        excla_count = getNumExcla(text)

        new_row = [lexical_diversity_rounded, char_count_rounded, link_count,
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




def run():
    delta_input = "/home/shared/CMV/SampledData/delta_sample_data.csv"
    nodelta_input = "/home/shared/CMV/SampledData/nodelta_sample_data.csv"
    word_list_input = "/home/shared/CMV/FeatureData/word_list.csv"

    output_delta = "/home/shared/CMV/FeatureData/delta_sample_feature_data.csv"
    output_nodelta = "/home/shared/CMV/FeatureData/nodelta_sample_feature_data.csv"

    delta_features, nodelta_features = generateFeatures([delta_input, nodelta_input], [output_delta, output_nodelta], word_list_input, 'con')


    delta_features = pd.DataFrame(data=delta_features, columns=None)
    nodelta_features = pd.DataFrame(data=nodelta_features, columns=None)


    print("Writing Features to File")
    delta_features.to_csv(output_delta, index=False)
    nodelta_features.to_csv(output_nodelta, index=False)
