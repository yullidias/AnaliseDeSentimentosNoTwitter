import os
import pandas as pd
import nltk
import pandas as pd
import numpy as np
import collections
import random
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')

def lerRootDir():
    try:
        with open('caminhoDiretorioTweets.txt', 'r') as f:
            rootDir = f.read() 
        if(rootDir.endswith('\n')):
            rootDir = rootDir[:-1]
        return rootDir
    except Exception as e:
        print(e)
        return None

def removeAccents(words):
    return [normalize('NFKD', word).encode('ASCII', 'ignore').decode('ASCII') for word in words]

def removePunctuation(words):
    tokenizePalavras = RegexpTokenizer(r'\w+')
    return [word for word in words if tokenizePalavras.tokenize(word)]

def removerStopWords(words):
    stop_words = removeAccents(set(stopwords.words('portuguese')))
    return [word for word in words if not word in stop_words]

def tokenize(text):
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(text)

def preprocessarTexto(text):
    words = tokenize(text.lower())
    words = removePunctuation(words)
    words = removeAccents(words)
    return " ".join(words)

def CriaBaseDeDados():
    base = pd.DataFrame()
    rootDir = lerRootDir()
    if rootDir != None:
        for file in os.listdir(rootDir):
            rootDir = rootDir if rootDir.endswith('/') else rootDir + '/'
            tweetsPD = pd.ExcelFile(rootDir + file).parse()
            base = base.append(tweetsPD)
        return base
    else:
        print("Falha ao ler diretório raiz")
        return None
    
def PreprocessarABase(base):
    tweetsProcessados = {"TweetProcessado" : []}
    for indice, linha  in base.iterrows():
        tweetsProcessados["TweetProcessado"] += [{
                "Tweet" : preprocessarTexto(str(linha["Tweet"])),
                "Polaridade" : linha["Polaridade"]}]
    return tweetsProcessados

def gerarVocabulario(base):
    vocabulario = [ linha["Tweet"] for linha in base["TweetProcessado"]]
    vectorizer = CountVectorizer()
    vectorizer.fit(vocabulario)#.todense()
    return vectorizer.vocabulary_

def criaTreino(base, porcentagem=0.70):
    indexTreino = set()
    while len(indexTreino) < (porcentagem * len(base["TweetProcessado"])):
        indexTreino.add(random.randint(0, len(base["TweetProcessado"]) - 1))
    return indexTreino

def criaTeste(treinoIndexs, base):
    teste = set()
    for i in range(len(base["TweetProcessado"])):
        if i not in treinoIndexs:
            teste.add(i)
    return teste

def criaEntradasMetodo(base, vocabulario, indexSet):
    feature = []
    saida = []

    for index in indexSet:
        vetorFrequencia = np.zeros(len(vocabulario))
        dictTweet = base["TweetProcessado"][index]
        words = tokenize(dictTweet["Tweet"])
        bow = collections.Counter(words)
        for word in words:
            index = vocabulario.get(word, "")
            if index != "":
                vetorFrequencia[int(index)] = bow[word]
        feature += [vetorFrequencia]
        saida += [dictTweet["Polaridade"]]
    
    return (feature, saida)

base = PreprocessarABase(CriaBaseDeDados())
vocabulario = gerarVocabulario(base)

treino = criaTreino(base)
teste = criaTeste(treino, base)

entrada = criaEntradasMetodo(base, vocabulario, treino)

X = entrada[0]
Y = entrada[1]

print("OK")
