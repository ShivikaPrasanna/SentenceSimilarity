from __future__ import division
from nltk import word_tokenize as tokenizer
from nltk.corpus import wordnet as wNet
from nltk.corpus import brown
from bestSense import bestSense
from compareSynset import pathLen, scalingDepthEffect

import math
import sys

#initialize wordCount to 0 and set dictionary to empty
wordCnt = 0
brownDict = {}

#Find the best word
def idealWord(wordA, wordSet):
    
    similarity = -1.0
    word = ""
    for wordB in wordSet:
        tempWord = semanticSimWords(wordA, wordB)
        if tempWord > similarity:
            similarity = tempWord
            word = wordB
    return word, similarity

#Check if dictionary has the word    
def content(wordData):
    
    global wordCnt
    if wordCnt == 0:
        for sentence in brown.sents():
            for word in sentence:
                key = word.lower()
                if not brownDict.has_key(key):
                    brownDict[key] = 0
                brownDict[key] += 1
                wordCnt += 1
    wordData = wordData.lower()
    if not brownDict.has_key(wordData):
        n = 0
    else:
        n = brownDict[wordData]
     
    #Calculate the Information Content (IC)   
    IC = math.log(n+1)/math.log(wordCnt+1)
    return 1.0 - IC
    
#Find words that are semantically similar
def semanticSimWords(wordA, wordB):
    idealPair = bestSense(wordA, wordB)
    shortestPathLen = pathLen(idealPair[0], idealPair[1])
    subsumerDepth = scalingDepthEffect(idealPair[0], idealPair[1])
    return shortestPathLen * subsumerDepth
     