#!/usr/bin/env python
# coding: utf-8

# In[23]:


import getpass
user_id = getpass.getuser()

import numpy as np
import pandas as pd
from patient_reps import *


# In[24]:


np.random.seed(0)
data_path = f"/home/{user_id}/OTTEHR/embedding/convae_architecture/data"


# In[42]:


low_code = 0
high_code = 100

def gen_codes(num_samples, num_features):    
    codes = np.random.randint(low_code, high_code, size=(num_samples, num_features))
    codes_list = [codes[i, :] for i in range(num_samples)]

    ids = np.array(range(1, num_samples+1))

    # id_codes = np.hstack((ids, codes))
    id_codes_df = pd.DataFrame({'MRN': ids, 'ENCODED-AVG':codes_list})
    id_codes_df['ENCODED-AVG'] = id_codes_df['ENCODED-AVG'].apply(lambda x: ','.join(map(str, x)))


    return ids, codes


# num_samples = 100
# num_features = 1000
# df = gen_codes(num_samples, num_features)


# In[43]:


num_samples = 100
num_features = 1000
ids, codes = gen_codes(num_samples, num_features)
input_file = "code.csv"
output_file = os.path.join(data_path, input_file)
with open(output_file, 'w') as f:
    wr = csv.writer(f)
    wr.writerow(["MRN", "ENCODED-AVG"])
    for m, e in zip(ids, codes):
        wr.writerow([m] + list(e))


# In[44]:


""" 
Generate vocabulary file
"""

def gen_vocab(vocabs, indir):
    """ 
    Generate vocabulary file to convae_architecture format

    :param list vocab: list of vocabularies
    :param string indir: directory to save vocab.csv
    """

    # generating vocabulary file
    print("generating vocabulary file")
    index_col = ["LABEL", "CODE"]
    mrn = []
    for code in vocabs:
        mrn.append("pat_"+str(code))

    outfile = os.path.join(indir, 'vocab.csv')
    print("vocabulary file path is:", outfile)
    with open(outfile, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(index_col)
        for m, c in zip(mrn, vocabs):
            wr.writerow([m, c])


# In[45]:


gen_vocab(list(range(low_code, high_code+1)), data_path)


# In[46]:


output_file = "code_emb.csv"
learn_patient_representations(data_path, input_file, output_file)


# In[ ]:




