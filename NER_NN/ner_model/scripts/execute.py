import utilities as ut
import vocabularyBuilder as vb
import argparse
#from rnn_seq import Model

'''
from configparser import ConfigParser

class Executor:
    def __init__(self,config_file):
        self.config_file = config_file
        self.parser = ConfigParser()
        self.parser.read(config_file)
'''

#def get_filenames(self):


word2vec_file = './Kannada/kan.word2vec'
tag_dict_file = './Kannada/tags'
train_file = './Kannada/kan.train1'
tune_file = './Kannada/kan.val1'
test_file = './Kannada/kan.test1'
char_vocab = './Kannada/kan.char'

characterVocabulary, reverseCharacterVocabulary, characterVocabularySize=vb.readCharacterVocabulary(char_vocab)
print('Read ',characterVocabularySize,' characters')

tagVocabulary, tagReverseVocabulary, numberOfTags=vb.readTagList(tag_dict_file)
print('Read ', numberOfTags, ' NER tags')
#One hot encoded vectors
tagVocabularyVector=ut.one_hot_encoder(tagVocabulary)

sourceDictionary, reverseSourceDictionary, sourceDictionarySize, embeddings, embeddingDimension=vb.loadEmbeddings(word2vec_file)
print("Read words with word embedding dimension ",embeddingDimension," and number of entries ",sourceDictionarySize)
#print(sourceDictionary)

train_sentences, train_max_seq_length = ut.readfile(train_file,tagVocabulary)
print("Training data contains ",len(train_sentences)," Lines "+" maximum number of words in a sentence is ",train_max_seq_length)


tune_sentences, tune_max_seq_length = ut.readfile(tune_file,tagVocabulary)
print("Development data contains ",len(tune_sentences)," Lines "," maximum number of words in a sentence is ",tune_max_seq_length)

test_sentences, test_max_seq_length = ut.readfile(test_file,tagVocabulary)
print("Test data contains ",len(test_sentences)," Lines "," maximum number of words in a sentence is ",test_max_seq_length)


def test_output():
    pass

def get_train_data():
    word_embed,tags = ut.get_nn_input(embeddings,embeddingDimension,train_sentences,train_max_seq_length,
                                      tagVocabularyVector)
    return word_embed, tags

def get_tune_data():
    word_embed,tags = ut.get_nn_input(embeddings,embeddingDimension,tune_sentences,tune_max_seq_length,
                                      tagVocabularyVector)
    return word_embed, tags

def get_test_data():
    word_embed,tags = ut.get_nn_input(embeddings,embeddingDimension,test_sentences,test_max_seq_length,
                                      tagVocabularyVector)
    return word_embed, tags


def get_train_rnn_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_dim', type=int, help='dimension of word vector', required=False, default=embeddingDimension)
    parser.add_argument('--sentence_length', type=int, help='max sentence length', required=False, default=train_max_seq_length)
    parser.add_argument('--class_size', type=int, help='number of classes', required=False, default=numberOfTags)
    parser.add_argument('--rnn_size', type=int, default=256, help='hidden dimension of rnn')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--restore', type=str, default=None, help="path of saved model")
    return parser
#--------------------------------------------
#Testing functions
emb,tags=get_train_data()
print("Dimension of emn vector",len(emb),len(emb[0]),len(emb[0][0]))
print('Dimension of tag vector',len(tags),len(tags[0]),len(tags[0][0]))

#---------------------------------------


def main():
    test_output()
    emb,tags=get_train_data()

if __name__=='__main__':
    main()













