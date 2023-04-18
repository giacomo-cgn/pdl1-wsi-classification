import pandas as pd
import numpy as np

class Data_ext:
    def __init__(self, data, results_dict, num_positives, num_negatives, batch1, batch2, batch3):
        self.data = data
        self.results_dict = results_dict
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.batch1 = batch1
        self.batch2 = batch2
        self.batch3 = batch3
        
def define_ext_dataset():
    #gets element from excel results file
    data = []
    df = pd.read_excel('../WSI/ext_wsi/PD-L1 AOUP.xlsx', header=0, usecols=['SLIDE','PD-L1'])
    
    positives = df.loc[df['PD-L1'] == 'pos'].loc[:,'SLIDE'].values.flatten()
    negatives = df.loc[df['PD-L1'] == 'neg'].loc[:,'SLIDE'].values.flatten()
             
    num_positives = len(positives)
    num_negatives = len(negatives)
    
    results_dict = {}
    for elem in positives:
        elem_cut = elem[:-4]
        results_dict[elem_cut] = 1
        data.append(elem_cut)
    
    for elem in negatives:
        elem_cut = elem[:-4]
        results_dict[elem_cut] = 0
        data.append(elem_cut)
        

    #first batch
    batch1 = [
    'M-6052-20 A1-SP142',
    'M-5697-20 SP142',
    'M-5696-20 (3)',
    'M-5428-20 SP142',
    'M-5248-20 SP142 (1)',
    'M-4092-20 SP142 (2)',
    'M-4075-20 SP142 (2)',
    'M-3914-20 SP142 (2)',
    'M-3905-20 SP142 (1)',
    'M-180-20 (3) B1',
    ]
    
    #second batch
    batch2 = [
        'M-3760-20-A1 SP142',
        'M-3587-20 SP142',
        'M-3454-20-A1 SP142',
        'M-1242-20 SP142',
        'M-1163-20 SP142',    
    ]
    
    #third batch
    batch3 = [
        'M-834-21 SP142',
        'M-534-20-1 SP142',
        'M-527-20 MA2-SP142',
        'M-527-20 MA1-SP142',
        'M-360-20 SP142',
        'M-313-20 SP142',
        'M-182-20-A1 SP142',
        'M-180-20-A1 SP142', #?
        'M-67-21-1-SP142 (4)',
        'M-3784-20-A1 SP142',
    ]

    # Temporary
    #data.remove('M-6014-20 A1-SP142')

                
    return Data_ext(data, results_dict, num_positives, num_negatives, batch1, batch2, batch3)