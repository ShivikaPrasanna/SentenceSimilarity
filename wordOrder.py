from __future__ import division

import math
import sys
import numpy as np
from nltk.corpus import brown
from nltk import tokenize
from sentenceSimilarity import idealWord

def wordOrder(words, jointWords, index):
    vector = np.zeros(len(jointWords))
    i = 0
    wordSet = set(words)
    for jointWord in wordSet:
        if jointWord in wordSet:
            vector[i] = index[jointWord]
        else:
            wordSim, sim = idealWord(jointWord, wordSet)
            if sim > 0.4:
                vector[i] = index[wordSim]
            else:
                vector[i] = 0
        i += 1
    return vector




