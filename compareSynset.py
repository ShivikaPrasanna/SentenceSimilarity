from __future__ import division

import sys
import math
from nltk.corpus import wordnet as wNet

def pathLen(synsetA, synsetB):
    maxLen = sys.maxint
    
    if synsetA is None or synsetB is None:
        return 0.0
    if synsetA == synsetB:
        maxLen = 0.0
    else:
        lemmaA = set([str(word.name()) for word in synsetA.lemmas()])
        lemmaB = set([str(word.name()) for word in synsetB.lemmas()])

        if len(lemmaA.intersection(lemmaB)) > 0:
            maxLen = 1.0
        else:
            maxLen = synsetA.shortest_path_distance(synsetB)
            if maxLen is None:
                maxLen = 0.0
    
    #alpha (a) = 0.2
    shortDist = math.exp(-0.2 * maxLen)
    return shortDist

def scalingDepthEffect(synsetA, synsetB):
    maxLen = sys.maxint
    #beta (b) = 0.45
    smoothingFactor = 0.45
    
    if synsetA is None or synsetB is None:
        return maxLen
    if synsetA == synsetB:
        maxLen = max(word[1] for word in synsetA.hypernym_distances())
    else:
        hypernymA = {word[0]: word[1] for word in synsetA.hypernym_distances()}
        hypernymB = {word[0]: word[1] for word in synsetA.hypernym_distances()}
        commonHypernymSet = set(hypernymA.keys()).intersection(set(hypernymB.keys()))
        if len(commonHypernymSet) <= 0:
            maxLen = 0
        else:
            distances = []
            for commonSet in commonHypernymSet:
                commonDistA = 0
                commonDistB = 0
                if hypernymA.has_key(commonSet):
                    commonDistA = hypernymA[commonSet]
                if hypernymB.has_key(commonSet):
                    commonDistBB = hypernymB[commonSet]
                
                maxDist = max(commonDistA, commonDistB)
                distances.append(maxDist)
            maxLen = max(distances)
            
    depthFunction = (math.exp(smoothingFactor * maxLen) - math.exp(-smoothingFactor * maxLen)) / (math.exp(smoothingFactor * maxLen) + math.exp(-smoothingFactor * maxLen))
        
    return depthFunction