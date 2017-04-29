import sys
from sklearn.metrics import f1_score,classification_report
import pandas as pd
import csv
import numpy as np



def calculateF1Score(crf_output_file):
        crf_output = pd.read_csv(crf_output_file,sep = '\t',header=None,quoting=csv.QUOTE_NONE,names= ['word','actual','predicted'])
        crf_target = crf_output[['actual','predicted']]
        crf_target.apply(lambda x: x.astype(str).str.upper())
        crf_target["actual"] = crf_target["actual"].str.replace("B-", "")
        crf_target["actual"] = crf_target["actual"].str.replace("I-", "")
        crf_target["predicted"] = crf_target["predicted"].str.replace("B-", "")
        crf_target["predicted"] = crf_target["predicted"].str.replace("I-", "")
        print(classification_report(crf_target['actual'], crf_target['predicted'], digits=4	))
        return f1_score(crf_target['actual'], crf_target['predicted'], average='macro')


input1=sys.argv[1]

print(np.mean(calculateF1Score(input1)))



