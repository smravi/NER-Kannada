import numpy as np

def one_hot_encoder(tagvocab):
    no_of_tags=len(tagvocab)
    tag_vector_dict={}
    for tag in tagvocab:
        tag_vector = np.zeros(no_of_tags,dtype=np.int)
        tag_vector[tagvocab[tag]]=1
        tag_vector_dict[tagvocab[tag]]=tag_vector
        print(tag,tag_vector)
    return tag_vector_dict

def readfile(fileName,tagVocab):
    lines = []
    with open(fileName, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    wordIndex=0
    charIndex=1
    tagIndex=2

    sourceSentences = []
    sourcewords=[]
    sourcetags=[]

    maxSequenceLength = 0
    sentence=[]

    for line in lines:
        if line in ['\n', '\r\n']:
            sourceSentences.append(sentence)
            if(len(sentence)>maxSequenceLength):
                maxSequenceLength=len(sentence)
            sentence=[]
        else:
            line = line.strip()
            linewords=line.split('\t')
            word=linewords[wordIndex]
            tag=tagVocab[linewords[tagIndex].upper()]
            characters = linewords[charIndex].strip().split()
            line=[word,characters,tag]
            sentence.append(line)
            sourcewords.append(word)
            sourcetags.append(tag)
    return sourceSentences,maxSequenceLength

def get_nn_input(embeddings,embedDimension, sentences,max_sentence_length,tag_vocab_vector):
    input_vectors=[]
    input_tags=[]
    wordvector=None
    for sentence in sentences:
        sentence_vector = []
        sentence_tags = []
        sentence_length=len(sentence)
        for line in sentence:
            word=line[0]
            tag_index=line[2]
            tag_vector=tag_vocab_vector[tag_index]
            if(embeddings.get(word) is None):
                wordvector=embeddings["</s>"]
            else:
                wordvector=embeddings[word]
            sentence_vector.append(wordvector)
            sentence_tags.append(tag_vector)
        for _ in range(max_sentence_length - sentence_length):
            temp = np.array([0 for _ in range(embedDimension)])
            sentence_vector.append(temp)
            sentence_tags.append(np.array([0] * len(tag_vocab_vector)))
        input_vectors.append(sentence_vector)
        input_tags.append(sentence_tags)

    return input_vectors,input_tags




