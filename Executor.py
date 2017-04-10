from configparser import SafeConfigParser
import sys,subprocess
import indicnlp
from indicnlp import common
from indicnlp.morph import unsupervised_morph
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import krippendorff_alpha

class Executor:
    def __init__(self,config_file):
        self.config_file = config_file
        parser = SafeConfigParser()
        parser.read(config_file)       
    
    def trainNER(self,crf_template_file,train_crf_file, crf_model_file):
        #findPOSTags()
        #findMorphenes()
        crf_train_process = subprocess.call('crf_learn ' + crf_template_file + ' ' + train_crf_file + ' ' + crf_model_file, shell= True)
        

    def testNER(self,test_crf_file, predicted_crf_output_file):
        crf_test_process = subprocess.call('crf_test ' + '-m ' + test_crf_file,shell=True, stdout = open(predicted_crf_output_file,'w'))
        
    
    def findPOSTags(self):
        tagger_module_path = parser.get('pos_tagger','pos_tagger_module_path')
        input_file = parser.get('words','input_word_file')
        tagger_output_file = parser.get('pos_tagger','pos_tagger_output') 
        print("Executing POS Tagger")
        process1 = subprocess.call(tagger_module_path + "/bin/tnt -H -v0 " + tagger_module_path + "/models/kannada " + input_file + "| sed -e 's/\t\+/\t/g' | " + tagger_module_path + "/bin/lemmatiser.py " + tagger_module_path + "/models/kannada.lemma" + ">" + "./raw_tagger_output_file",shell=True, stdout = open('pos_tag.log','w'))
        pos_tag_fp = open(tagger_output_file,'w')
        #pos_tag_fp.write('Word' + '  '+ 'POSTag' + '\n')
        with open('./raw_tagger_output_file','r',encoding='utf-8') as fp:
            for line in fp:
                if line != '\n':
                    [word, tag] = line.strip().split()[0:2]
                    processed_pos_tag = tag.strip().split('.')[0]
                    pos_tag_fp.write(word.strip() + ' ' + processed_pos_tag.strip() + "\n")
        pos_tag_fp.close()
        print("POS Tagger Execution Finished")
     
    def findMorphenes(self):
        input_file = parser.get('words','input_word_file')
        morpheme_output_file = parser.get('morphessor','morpheme_output_file')
        print("Executing Morpheme Analyzer")
        morpheme_analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer('kn')
        input_fptr = open(input_file,'r',encoding = 'utf-8')
        output_fptr = open(morpheme_output_file,'w',encoding='utf-8')
        for line in input_fptr:
            word = line.strip().split(' ')
            if len(word) != 0: 
                morpheme_tokens = morpheme_analyzer.morph_analyze_document(word)
                concatenated_tokens = ' '.join(' '.join(str(x) for x in w) for w in morpheme_tokens)
                if len(concatenated_tokens.split()) == 0:
                    concatenated_tokens += '0 0'
                if len(concatenated_tokens.split()) == 1:
                    concatenated_tokens += ' 0'
                concatenated_tokens = line.strip() + ' ' + concatenated_tokens
                tokens = concatenated_tokens.split(' ') 
                output_fptr.write(tokens[1] + ' ' + tokens[-1])
                output_fptr.write('\n')
        print("Morpheme Analyzer Execution Finished")
    
    def mergeModuleOutputs(self, pos_tagger_output_file, morpheme_output_file, ner_tags_file, merged_output_file ):
        pos_tags_dataframe = pd.read_csv(pos_tagger_output_file, sep =' ',header = None,names = ['word','postag'])
        morpheme_tags = pd.read_csv(morpheme_output_file,sep=' ',header=None,names = ['stem','suffix'])
        ner_tags_dataframe = pd.read_csv(ner_tags_file, header = None,sep =' ',names=['nertag'])
        merged_data = pd.concat([pos_tags_dataframe,ner_tags_dataframe,morpheme_tags],axis = 1)
        final_dataframe = merged_data[['word','postag','stem','suffix','nertag']]
        final_dataframe.to_csv(merged_output_file, sep = ' ', header= None, index= False)
        
    
    def calculateF1Score(self,crf_output_file):
        crf_output = pd.read_csv(crf_output_file,sep = '\t',header=None,names= ['word','postag','stem','suffix','actual','predicted']) 
        crf_target = crf_output[['actual','predicted']]
        crf_target.apply(lambda x: x.astype(str).str.upper())
        return f1_score(crf_target['actual'], crf_target['predicted'], average='micro')
   
    def calculateKripendorffCoeeficient(input_annotator_file):
        missing = '*'
        input_df = pd.DataFrame(input_annotator_file,sep = ' ',header = None,names = ['word','ann1','ann2'])
        input_array = input_df.values.tolist()
        print("nominal metric: %.3f" % krippendorff_alpha(array, nominal_metric, missing_items=missing))
        print("interval metric: %.3f" % krippendorff_alpha(array, interval_metric, missing_items=missing))
        
                
if __name__ =='__main__':
    parser = SafeConfigParser()
    config_file = sys.argv[1]
    parser.read(config_file)
    common.set_resources_path(parser.get('indic_config','indic_resource_path'))
    NER_executor = Executor(config_file)
    NER_executor.findPOSTags()
    NER_executor.findMorphenes()
    NER_executor.mergeModuleOutputs(parser.get('pos_tagger','pos_tagger_output'), "./morphemes_train_ner_tags.out", parser.get('ner_tag_data','ner_word_tags'),parser.get('crf_learner','crf_input_file'))
    NER_executor.trainNER(parser.get(crf_learner,crf_template_file),parser.get(crf_learner,train_crf_file),parser.get(crf_learner, crf_model_file))
    print(NER_executor.calculateF1Score("./crf_output.txt"))
    

