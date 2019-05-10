from __future__ import division

import math
import sys
import numpy as np

from nltk.corpus import brown
from nltk import tokenize
from sentenceSimilarity import idealWord, content

def semanticVector(words, jointWords, normalizedIC):
    i = 0
    wordSet = set(words)
    vector = np.zeros(len(jointWords))
    for jointWord in jointWords:
        if jointWord in wordSet:
            vector[i] = 1
            if normalizedIC:
                vector[i] = vector[i] * math.pow(content(jointWord), 2)
        else:
            simWord, similarity = idealWord(jointWord, wordSet)
            if similarity > 0.2:
                vector[i] = 0.2
            else:
                vector[i] = 0.0
            if normalizedIC:
                vector[i] = vector[i] * content(jointWord) * content(simWord)
        i += 1
    
    return vector
    
