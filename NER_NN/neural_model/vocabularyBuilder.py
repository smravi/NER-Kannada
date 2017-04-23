import numpy as np


def readCharacterVocabulary(filename):
    characterVocabulary={}
    characterReverseVocabulary={}
    characterIndex=0

    lines=[]
    with open(filename,'r',encoding='utf-8') as fp:
        lines=fp.readlines()

    for line in lines:
        currentLength=0
        for word in line.strip(' \t'):
            if word != ' ':
                if (characterVocabulary.get(word) is None):
                    characterVocabulary[word] = characterIndex
                    characterReverseVocabulary[word] = word
                    characterIndex = characterIndex + 1

    vocabularySize = 0
    characterVocabulary["</S>"] = characterIndex
    characterReverseVocabulary[characterIndex] = "</S>"

    characterIndex = characterIndex + 1
    characterVocabulary["<S>"] = characterIndex
    characterReverseVocabulary[characterIndex] = "<S>"

    for _ in characterVocabulary:
        vocabularySize = vocabularySize + 1

    print('Read ',vocabularySize,' characters')
    return characterVocabulary, characterReverseVocabulary, vocabularySize

def readTagList(fileName):
    vocabulary={}
    reverseVocabulary={}
    index=0

    lines=[]
    with open(fileName,'r',encoding='utf-8') as fp:
        lines=fp.readlines()

    for line in lines:
        if (vocabulary.get(line) is None):
            line=line.strip('\n').upper()
            vocabulary[line] = index
            reverseVocabulary[index] = line
            index = index + 1

    vocabularySize = 0
    for i, v in enumerate(vocabulary):
        vocabularySize = vocabularySize + 1
        print(i,v)
    return vocabulary, reverseVocabulary, vocabularySize

def loadEmbeddings(fileName):
    lines = []
    with open(fileName, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    numberOfLines = 0
    vocabulary = {}
    reverseVocabulary={}
    embeddings = {}

    metadata = lines[0].split()
    dimension = int(metadata[1])
    print('Dimension',dimension)

    for line in lines[1:]:
        #wordvector = np.array([1,1]) #default initialization
        words = line.split()
        word = words[0]
        #wordvector = [float(w) for w in words[1:]]
        wordvector = np.array([float(val) for val in words[1:]])
        #print(wordvector)

        if dimension!=len(wordvector):
            #print('Dimension', dimension)
            #print(len(wordvector))
            print('Skipped loading embedding for a word '+word)
        else:
            if(vocabulary.get(word) is None):
                vocabulary[word] = numberOfLines
                reverseVocabulary[numberOfLines] = word
            embeddings[word] = wordvector
            numberOfLines = numberOfLines + 1

    #vocabulary["<S>"] = numberOfLines
    #reverseVocabulary[numberOfLines] = "<S>"
    #embeddings["<S>"] = np.array([float(0) for _ in range(dimension)])

    vocabularySize = len(vocabulary)

    return vocabulary, reverseVocabulary, vocabularySize, embeddings, dimension










