import os
import pandas as pd
import nltk
import pandas as pd
import numpy as np
import collections
import random
from enum import Enum, auto
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer

#nltk.download('punkt')
#nltk.download('stopwords')

class Tag(Enum):
    DIGIT = 'DIGITO'
    MONEY = 'DINHEIRO'

def readRootDir():
    try:
        with open('caminhoDiretorioTweets.txt', 'r') as f:
            rootDir = f.read()
            rootDir = rootDir[:-1] if rootDir.endswith("\n") else rootDir
            return rootDir if rootDir.endswith('/') else rootDir + '/'
        return None
    except Exception as e:
        print(e)
        return None

def removeAccents(words):
    return [normalize('NFKD', word).encode('ASCII', 'ignore').decode('ASCII') for word in words]

def removePunctuation(words):
    tokenizeWords = RegexpTokenizer(r'\w+')
    return [word for word in words if tokenizeWords.tokenize(word)]

def removerStopWords(words):
    stopWords = removeAccents(set(stopwords.words('portuguese')))
    return [word for word in words if word not in stopWords]

def wordStemmer(words):
    stemmer = RSLPStemmer()
    return [stemmer.stem(word) for word in words]

def tagNumbers(words):
    for count in range(len(words)):
        if words[count].isdigit():
            if count > 0 and words[count - 1] == '$':
                if count > 1 and words[count - 2] == 'r':
                    words[count - 2] = ''
                    words[count - 1] = ''
                    words[count] = Tag.MONEY.value
                else:
                    words[count - 1] = ''
                    words[count] = Tag.MONEY.value
            elif (count + 1) < len(words) and words[count + 1] == ('reais' or 'real'):
                words[count : count + 2] = Tag.MONEY.value
            else:
                words[count] = Tag.DIGIT.value
    return words

def removeRepeatChar(words):
    new_words = []
    for word in words:
        new_word = ""
        for c in word:
            if(len(new_word)>1):
                if(new_word[-1]!=c or new_word[-2]!=c):
                    new_word+=c
            else:
                new_word+=c
        new_words.append(new_word)
    return new_words

def tokenize(text):
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(text)

def preprocessarTexto(text, isToRemoveStopWords, isToStemWords):
    words = tokenize(text.lower())
    words = removeAccents(words)
    words = tagNumbers(words)
    words = removePunctuation(words)
    words = removeRepeatChar(words)

    if isToRemoveStopWords:
        words = removerStopWords(words)
    if isToStemWords:
        words = wordStemmer(words)
    return " ".join(words)

def getDataFromFiles():
    base = pd.DataFrame()
    rootDir = readRootDir()
    if rootDir != None:
        for file in os.listdir(rootDir):
            tweetsPD = pd.ExcelFile(rootDir + file).parse()
            tweetsPD['Tweet'] = tweetsPD['Tweet'].values.astype('U')
            tweetsPD['Polaridade'] = tweetsPD['Polaridade'].values.astype('U')
            base = base.append(tweetsPD)
        return base
    else:
        print("Falha ao ler diretório raiz")
        return None

def preprocessBase(base, isToRemoveStopWords, isToStemWords):
    for index, line  in base.iterrows():
        line["Tweet"] = preprocessarTexto(str(line["Tweet"]), isToRemoveStopWords, isToStemWords)

def preprocess(porcentagemTreino=0.7, isToRemoveStopWords=False, isToStemWords=False):
    base = getDataFromFiles()
    preprocessBase(base, isToRemoveStopWords, isToStemWords)
    base = base.sample(frac=1) #random the tweets
    size = int(porcentagemTreino * len(base))
    training = base[0 : size]
    test = base[size : len(base)]

    vectorizer = CountVectorizer(analyzer='word')
    vectorizer.fit_transform(base["Tweet"])
    vocabularyTraining = vectorizer.transform(training["Tweet"])
    vocabularyTest = vectorizer.transform(test['Tweet'])
    del base
    return (vocabularyTraining, training, vocabularyTest, test)
