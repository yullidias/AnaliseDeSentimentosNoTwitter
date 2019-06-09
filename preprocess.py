import os
import pandas as pd
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
from enum import Enum, auto
import nltk
import re
import time
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer #conda install -c districtdatalabs yellowbrick
from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('rslp')

class Tag(Enum):
    DIGIT = 'digito'
    MONEY = 'dinheiro'
    EMAIL = 'email'
    URL = 'url'
    RISOS = 'RISOS'

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


def tagURL(text):
    return re.sub(r'https?:\/\/(www\.)?[0-9A-Za-z:%_\+.~#?&//=]+[^\s]{2,4}(\/[0-9A-Za-z:%_\+.~#?&//=]+)?', Tag.URL.value, text)

def tagEmail(text):
    return re.sub(r'[A-za-z0-9-._]+@[A-za-z]+\.[^\s]+', Tag.EMAIL.value, text)

def othersChanges(words):
    for count in range(len(words)):
        if words[count] == 'gnt':
            words[count] = 'gente'
        if words[count] == 'jnts':
            words[count] = 'juntos'
        if words[count] == 'q':
            words[count] = 'que'
        if words[count] == 'p':
            words[count] = 'para'
        if words[count] == 'pra':
            words[count] = 'para'
        if words[count] == 'c':
            words[count] = 'com'
        if words[count] == 'n':
            words[count] = 'nao'
        if words[count] == 'vc':
            words[count] = 'voce'
        if words[count] == 's':
            words[count] = 'sem'
        if words[count] == 'd':
            words[count] = 'de'
        if words[count] == 'brazil':
            words[count] = 'brasil'
        if words[count] == 'hj':
            words[count] = 'hoje'
        if words[count] == 'haha' or words[count] == 'kk':
            words[count] = Tag.RISOS.value
    return words

def preprocessBeforeTokenize(text):
    text = tagURL(text)
    text = tagEmail(text)
    text = tagNumbers(text)
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
    words = removePunctuation(words)
    words = removeRepeatChar(words)
    words = othersChanges(words)

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

def plotData(base, labels=[-1,0,1]):
    vectorizer = CountVectorizer(lowercase=False)
    tweets       = vectorizer.fit_transform(base['Tweet'])
    tsne = TSNEVisualizer()
    tsne.fit(tweets, labels)
    tsne.poof()

def plotMostFrequentWords(base):
    vectorizer = CountVectorizer(lowercase=False, max_df=0.7)
    docs       = vectorizer.fit_transform(base['Tweet'])
    features   = vectorizer.get_feature_names()
    visualizer = FreqDistVisualizer(features=features)
    visualizer.fit(docs)
    visualizer.poof()

def getBalencedTrainFromHeadBase(base, class1, class2, sizeTraining):
    base = base.head(sizeTraining)
    resultfeature1 = base.loc[base['Polaridade'] == 1 ]
    resultfeature2 = base.loc[base['Polaridade'] == -1 ]
    resultfeature3 = base.loc[base['Polaridade'] == 0 ]

    sizeByFeature = int(sizeTraining / 3)
    minSize = min(len(resultfeature1), len(resultfeature2), len(resultfeature3))
    size = minSize if minSize < sizeByFeature else sizeByFeature

    result =               resultfeature1.head(size)
    result = result.append(resultfeature2.head(size))
    result = result.append(resultfeature3.head(size))
    return result

def preprocess(porcentagemTreino=0.7, isToRemoveStopWords=False, isToStemWords=False):
    base = getDataFromFiles()
    preprocessBase(base, isToRemoveStopWords, isToStemWords)
    preprocessPolaridade(base)
    base.sample(frac=1, replace=True) #random the tweets

    sizeTraining = int(porcentagemTreino * len(base))

    training = getBalencedTrainFromHeadBase(base, 1, -1, sizeTraining)
    test = base.tail(len(base) - sizeTraining)

    plotMostFrequentWords(training)
    #min_df : float in range [0.0, 1.0] or int, default=1
    #When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
    vectorizer = CountVectorizer(lowercase=False, max_features=4000)
    vectorizer.fit_transform(base["Tweet"])
    vocabularyTraining = vectorizer.transform(training["Tweet"])
    vocabularyTest = vectorizer.transform(test["Tweet"])

    del base
    return (vocabularyTraining, training, vocabularyTest, test)
