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
    punc = '\"\\\'/;:,.!?\n'
    for char in text:
        if (char in punc):
            text = text.replace(char, " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text
