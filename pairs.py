from similarityType import sentenceSim
from bestSense import bestSense
from sentenceSimilarity import semanticSimWords

#Author: Shivika Prasanna
#Course: Natural Language Processing, Spring 2018
#Purpose: Implementation of Sentence Similarity based on Semantic Nets and Corpus Statistics


#wordPairs as in the paper, arrange alphabetically by the first distinct Noun, else the second noun in the pair
wordPairs = [
    ["asylum", "fruit", 0.01],
    ["autograph", "shore", 0.01],
    ["autograph", "signature", 0.41],
    ["automobile", "car", 0.56],
    ["bird", "woodland", 0.01],
    ["boy", "lad", 0.58],
    ["boy", "rooster", 0.11],
    ["boy", "sage", 0.04],
    ["cemetery", "graveyard", 0.77],
    ["coast", "forest", 0.13],
    ["coast", "shore", 0.59],
    ["cock", "rooster", 0.86],
    ["cord", "smile", 0.01],
    ["cord", "string", 0.47],
    ["cushion", "pillow", 0.52],
    ["forest", "graveyard", 0.07],
    ["forest", "woodland", 0.63],
    ["furnace", "stove", 0.35],
    ["gem", "jewel", 0.65],
    ["glass", "tumbler", 0.14],
    ["grin", "smile", 0.49],
    ["hill", "mound", 0.29],
    ["hill", "woodland", 0.15],
    ["implement", "tool", 0.59],
    ["journey", "voyage", 0.36],
    ["magician", "oracle", 0.13],
    ["magician", "wizard", 0.36],
    ["midday", "noon", 0.96],
    ["oracle", "sage", 0.28],
    ["serf", "slave", 0.48]
]

sentencePairs = [
    ["I like that bachelor.", "I like that unmarried man.", 0.561],
    ["John is very nice.", "Is John very nice?", 0.977],
    ["It is a dog.", "That must be your dog.", 0.739],
    ["It is a dog.", "It is a log.", 0.623],
    ["It is a dog.", "It is a pig.", 0.790],
    ["I have a hammer.", "Take some nails.", 0.508],
    ["I have a pen.", "Where is ink.", 0.129],
    ["A glass of cider.", "A full cup of apple juice.", 0.678],
    ["I have a pen.", "Where do you live?", 0.0],
    ["Red alcoholic drink.", "A bottle of wine.", 0.585],
    ["Red alcoholic drink.", "Fresh orange juice.", 0.611],
    ["Red alcoholic drink.", "An English dictionary.", 0.0],
    ["Dogs are animals.", "They are common pets.", 0.738],
    ["Canis familiaris are animals.", "Dogs are common pets.", 0.362],
    ["Red alcoholic drink.", "Fresh apple juice.", 0.420],
    ["I have a hammer.", "Take some apples", 0.121],
]

for wordPair in wordPairs:
    print "%s\t%s\t%.2f\t%.2f" % (wordPair[0], wordPair[1], wordPair[2], 
                                  semanticSimWords(wordPair[0], wordPair[1]))
                                  
for sentencePair in sentencePairs:
    print "%s\t%s\t%.3f\t%.3f\t%.3f" % (sentencePair[0], sentencePair[1], sentencePair[2], 
        sentenceSim(sentencePair[0], sentencePair[1], False),
        sentenceSim(sentencePair[0], sentencePair[1], True))
