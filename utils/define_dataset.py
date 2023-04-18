import pandas as pd
import numpy as np

class Data:
    def __init__(self, data, results_dict, test_set, tr_set, num_positives, num_negatives):
        self.data = data
        self.results_dict = results_dict
        self.test_set = test_set
        self.tr_set = tr_set
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        
def define_dataset():
    #gets element from excel results file
    data = []
    df_positives = pd.read_excel('../WSI/thesis_WSI/resultsWSI.ods', sheet_name='PD-L1 POSITIVE', header=None)
    df_negatives = pd.read_excel('../WSI/thesis_WSI/resultsWSI.ods', sheet_name='PD-L1 NEGATIVE', header=None)

    positives_arr = df_positives.iloc[:,[0]].values.flatten()
    negatives_arr = df_negatives.iloc[:,[0]].values.flatten()
             
    num_positives = 0
    num_negatives = 0
    
    results_dict = {}
    for elem in positives_arr:
        results_dict[elem] = 1
        data.append(elem)
        num_positives+=1
    for elem in negatives_arr:
        results_dict[elem] = 0
        data.append(elem)
        num_negatives+=1
        
 
    
    #define tr and test set
    test_set = [
    '20-COMP-077',
    '20-COMP-065',
    '19-COMP-039',
    '20-COMP-102',
    '19-COMP-037',
    '19-COMP-026'
    ]

    tr_set = []
    for elem in data:
        tr_set.append(elem)
        for test_el in test_set:
            if test_el==elem:
                tr_set.remove(elem)
                
                
    return Data(data, results_dict, test_set, tr_set, num_positives, num_negatives)