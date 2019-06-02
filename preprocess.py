import os
import pandas as pd
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
from enum import Enum, auto
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')

class Tag(Enum):
    DIGIT = 'digito'
    MONEY = 'dinheiro'
    EMAIL = 'email'
    URL = 'url'

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

def tagNumbers(text):
    text = re.sub(r'[0-9]+([.,]?[0-9]+)?', Tag.DIGIT.value, text)
    text = re.sub(r'r?\$[\s]*'+Tag.DIGIT.value, Tag.MONEY.value, text)
    return text

def tagDollar(text):
    return re.sub(r'\$', Tag.DOLLAR.value, text)

def tagURL(text):
    return re.sub(r'https?:\/\/(www\.)?[0-9A-Za-z:%_\+.~#?&//=]+[^\s]{2,4}(\/[0-9A-Za-z:%_\+.~#?&//=]+)?', Tag.URL.value, text)

def tagEmail(text):
    return re.sub(r'[A-za-z0-9-._]+@[A-za-z]+\.[^\s]+', Tag.EMAIL.value, text)

def preprocessBeforeTokenize(text):
    text = tagURL(text)
    text = tagEmail(text)
    text = tagNumbers(text)
    text = tagDollar(text)
    return text

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
    text = preprocessBeforeTokenize(text.lower())
    words = tokenize(text)
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
            tweetsPD['Polaridade'] = tweetsPD['Polaridade'].values.astype(int)
            base = base.append(tweetsPD)
            break
        return base
    else:
        print("Falha ao ler diretorio raiz")
        return None

def preprocessBase(base, isToRemoveStopWords, isToStemWords):
    for index, line  in base.iterrows():
        base.at[index,"Tweet"] = preprocessarTexto(str(line["Tweet"]), isToRemoveStopWords, isToStemWords)

def preprocessPolaridade(base):
    for index, line  in base.iterrows():
        if line['Polaridade']  < -1:
            base.at[index,'Polaridade'] = -1
        elif line['Polaridade']  > 1:
            base.at[index,'Polaridade'] = 1

def plotData(base):
    pos = base.loc[base['Polaridade'] == 1 ]
    neg = base.loc[base['Polaridade'] == -1 ]
    neu = base.loc[base['Polaridade'] == 0 ]
    plt.scatter(pos.index.values, pos['Polaridade'], s=60, c='k', marker='+', linewidths=1)
    plt.scatter(neg.index.values, neg['Polaridade'], s=60, c='k', marker='+', linewidths=1)
    plt.scatter(neu.index.values, neu['Polaridade'], s=60, c='k', marker='+', linewidths=1)
    plt.show()

def preprocess(porcentagemTreino=0.6, isToRemoveStopWords=False, isToStemWords=False):
    base = getDataFromFiles()
    preprocessBase(base, isToRemoveStopWords, isToStemWords)
    preprocessPolaridade(base)
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
