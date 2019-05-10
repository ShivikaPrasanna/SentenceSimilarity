from __future__ import division

import sys
from nltk.corpus import wordnet as wNet

def bestSense(wordA, wordB):
    sim = -1.0
    wordASynset = wNet.synsets(wordA)
    wordBSynset = wNet.synsets(wordB)
    
    if len(wordASynset) == 0 or len(wordBSynset) == 0:
        return None, None
    else:
        sim = -1.0
        idealPair = None, None
        
        for synsetA in wordASynset:
            for synsetB in wordBSynset:
                temp = wNet.path_similarity(synsetA, synsetB)
                if temp > sim:
                    sim = temp
                    idealPair = synsetA, synsetB
        return idealPair