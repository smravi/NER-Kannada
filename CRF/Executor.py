from configparser import SafeConfigParser
import sys,subprocess
import indicnlp
from indicnlp import common
from indicnlp.morph import unsupervised_morph
from sklearn.metrics import f1_score,classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from krippendorff_alpha import nominal_metric,krippendorff_alpha,interval_metric
import csv

class Executor:
    def __init__(self,config_file):
        self.config_file = config_file
        parser = SafeConfigParser()
        parser.read(config_file)       
    
    def trainNER(self):
        #findPOSTags()
        #findMorphenes()
        crf_template_file = parser.get('crf_learner','crf_template_file')
        train_crf_file = parser.get('crf_learner','train_crf_file')
        crf_model_file = parser.get('crf_learner','crf_model_file')
        crf_train_process = subprocess.call('crf_learn ' + crf_template_file + ' ' + train_crf_file + ' ' + crf_model_file, shell= True)
        

    def testNER(self):
        test_crf_file = parser.get('crf_test','crf_test_file')
        predicted_crf_output_file = parser.get('crf_test','crf_test_output_file')
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

    def find_suffix_features(self):
        print("Creating Suffix Features")
        person_input_file = open(parser.get('suffix_files','person_suffix_file'),'r',encoding = 'utf-8')
        location_input_file = open(parser.get('suffix_files','location_suffix_file'),'r',encoding = 'utf-8')
        org_input_file = open(parser.get('suffix_files','org_suffix_file'),'r',encoding = 'utf-8')
        input_file = open(parser.get('words','input_word_file'),'r',encoding = 'utf-8')
        output_fptr = open(parser.get('suffix_files', 'suffix_output_file'),'w',encoding='utf-8')
        person_suffices = []
        location_suffices = []
        org_suffices = []
        for person in person_input_file:
            person_suffices.append(person.strip())
        for location in location_input_file:
            location_suffices.append(location.strip())
        for org in person_input_file:
            org_suffices.append(org.strip())
        for word in input_file:
            if len(word) != 0:
                word = word.strip()
                output_features = ['N','N','N']
                for person in person_suffices:
                    if word.strip().endswith(person):
                        output_features[0]= 'Y'
                        break
                for location in location_suffices:
                    if word.strip().endswith(location):
                        output_features[1] ='Y'
                        break
                for org in org_suffices:
                    if word.strip().endswith(org):
                        output_features[2] = 'Y'
                        break
                output_fptr.write(' '.join(str(f) for f in output_features))
                output_fptr.write('\n')
        output_fptr.close()
        print("Suffix Features Creation Finished")
            
    def find_date_features(self):
        print("Creating Date Features")
        input_file = parser.get('words','input_word_file')
        df = pd.read_csv(input_file,sep ="\t",header=None)
        months_days = pd.read_csv(parser.get('suffix_files','month_days'),'r',encoding = 'utf-8',header=None)
        date_feature = df[0].str.contains('[1-9][0-9]{3}',regex= True) | df[0].str.contains('[\u0CE7-\u0CEF][\u0CE6-\u0CEF]{3}',regex = True) | df[0].isin(months_days[0].tolist())
        date_feature = date_feature.astype(int).apply(lambda x: 'Y' if x == 1 else 'N')
        print("Date Feature Created")
        return date_feature

    
    def mergeModuleOutputs(self, pos_tagger_output_file, morpheme_output_file, ner_tags_file, suffix_tags_file,merged_output_file):
        pos_tags_dataframe = pd.read_csv(pos_tagger_output_file, sep =' ',header = None,names = ['word','postag'])
        morpheme_tags = pd.read_csv(morpheme_output_file,sep=' ',header=None,names = ['stem','suffix'])
        ner_tags_dataframe = pd.read_csv(ner_tags_file, header = None,sep =' ',names=['nertag'])
        prefix_suffix_df = pd.read_csv(parser.get('suffix_files','prefix_suffix_file'),sep = '\t',encoding = 'utf-8',header=None,names=['firstword', 'Prefix3', 'Prefix4', 'Prefix5', 'Suffix3', 'Suffix4', 'Suffix5'])
        firstword_feature = prefix_suffix_df['firstword'].shift() == prefix_suffix_df['firstword']
        suffix_tags_file = pd.read_csv(suffix_tags_file,header = None, sep = ' ',names = ['person_suffix','location_suffix','org_suffix'])
        merged_data = pd.concat([pos_tags_dataframe,ner_tags_dataframe,morpheme_tags,prefix_suffix_df,suffix_tags_file],axis = 1)
        merged_data["date"] = self.find_date_features()
        merged_data["prevtag"] = merged_data["nertag"].shift(1).fillna('O')
        merged_data["prevprevtag"] = merged_data["nertag"].shift(2).fillna('O') 
        merged_data['firstwordbinary'] = firstword_feature.astype(int).apply(lambda x: 'Y' if x == 1 else 'N')
        print(merged_data.head)
        final_dataframe = merged_data[['word','postag','stem','suffix','date','firstwordbinary','firstword' ,'Prefix3', 'Prefix4', 'Prefix5', 'Suffix3', 'Suffix4', 'Suffix5','prevtag','prevprevtag','person_suffix','location_suffix','org_suffix','nertag']] #person_suffix','location_suffix','org_suffix','nertag']]
        final_dataframe.to_csv(merged_output_file, sep = ' ', header= None, index= False)
        
    
    def calculateF1Score(self,crf_output_file):
        crf_output = pd.read_csv(crf_output_file,sep = '\t',header=None,quoting=csv.QUOTE_NONE,names= ['word','postag','stem','suffix','date','firstwordbinary','firstword', 'Prefix3', 'Prefix4', 'Prefix5', 'Suffix3', 'Suffix4', 'Suffix5','prevtag','prevprevtag','person_suffix','location_suffix','org_suffix','actual','predicted']) 
        crf_target = crf_output[['actual','predicted']]
        crf_target.apply(lambda x: x.astype(str).str.upper())
        crf_target["actual"] = crf_target["actual"].str.replace("B-", "")
        crf_target["actual"] = crf_target["actual"].str.replace("I-", "")
        crf_target["predicted"] = crf_target["predicted"].str.replace("B-", "")
        crf_target["predicted"] = crf_target["predicted"].str.replace("I-", "")
        print(classification_report(crf_target['actual'], crf_target['predicted']))
        return f1_score(crf_target['actual'], crf_target['predicted'], average=None) #'micro')
   
    def calculateKripendorffCoeeficient(self,input_annotator_file):
        missing = '*'
        input_df = pd.read_csv(input_annotator_file,sep = '\t',quoting=csv.QUOTE_NONE,header = None,encoding= 'utf-8',names = ['word','ann1','ann2'])
        print("Checking")
        input_array_ann1 = input_df[['ann1']].values.tolist()
        input_array_ann2 = input_df[['ann2']].values.tolist()

        print("nominal metric: %.3f" % krippendorff_alpha([sum(input_array_ann1,[]),sum(input_array_ann2,[])], nominal_metric, missing_items=missing,convert_items=str))
        #print("interval metric: %.3f" % krippendorff_alpha(input_array, interval_metric, missing_items=missing,convert_items=str))
        
                
if __name__ =='__main__':
    parser = SafeConfigParser()
    config_file = sys.argv[1]
    parser.read(config_file)
    common.set_resources_path(parser.get('indic_config','indic_resource_path'))
    NER_executor = Executor(config_file)
    NER_executor.findPOSTags()
    NER_executor.findMorphenes()
    NER_executor.find_suffix_features()
    NER_executor.mergeModuleOutputs(parser.get('pos_tagger','pos_tagger_output'), parser.get('morphessor','morpheme_output_file'), parser.get('ner_tag_data','ner_word_tags'),parser.get('suffix_files','suffix_output_file'),parser.get('crf_learner','crf_input_file'))
    #NER_executor.trainNER()
#    print(NER_executor.calculateF1Score("./final_crf_output_1"))
    #NER_executor.calculateKripendorffCoeeficient("./interannotation")
    

