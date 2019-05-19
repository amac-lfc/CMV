# Lib

#optimize the loops so you only need to run through the text once

# these are the words of certainty
wOC = ["absolutely", "always", "certain", "commit", "completely", "every", "exact"]

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

def cleanText(text):
    len_of_text = len(text)
    print(f"Total num of chars: {len_of_text}")
    print_nums = [int(i/10 * len_of_text) for i in range(1,10)]
    count = 0
    punc = '\"\\/;:,.!?\n><()[]{}-'
    for char in text:
        if count in print_nums:
            print(f"{count*100//len_of_text + 1}% Complete")
        if (char in punc):
            text = text.replace(char, " ")
        count += 1
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
