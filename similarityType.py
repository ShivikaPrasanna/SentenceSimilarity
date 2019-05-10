from __future__ import division

import math
import sys

from nltk import word_tokenize as wordTokenizer
from wordOrder import wordOrder
from semanticVector import semanticVector
from numpy import dot, linalg

def semanticSim(sentA, sentB, normalizedIC):
    wordSetA = wordTokenizer(sentA)
    wordSetB = wordTokenizer(sentB)
    wordSet = set(wordSetA).union(set(wordSetB))
    wordVectorA = semanticVector(wordSetA, wordSet, normalizedIC)
    wordVectorB = semanticVector(wordSetB, wordSet, normalizedIC)
    
    semanticSimilarity = dot(wordVectorA, wordVectorB) / (linalg.norm(wordVectorA) * linalg.norm(wordVectorB))
    return semanticSimilarity
    
def wordSim(sentA, sentB):
    wordSetA = wordTokenizer(sentA)
    wordSetB = wordTokenizer(sentB)
    wordSet = list(set(wordSetA).union(set(wordSetB)))
    index = {word[1]: word[0] for word in enumerate(wordSet)}
    
    r1 = wordOrder(wordSetA, wordSet, index)
    r2 = wordOrder(wordSetB, wordSet, index)
    tempSr = linalg.norm(r1-r2)/linalg.norm(r1+r2)
    wordSimilarity = 1 - tempSr
    return wordSimilarity

def sentenceSim(sentA, sentB, normalizedIC):
    sentenceSimilarity = 0.58 * semanticSim(sentA, sentB, normalizedIC) + 0.42 * wordSim(sentA, sentB)
    return sentenceSimilarity