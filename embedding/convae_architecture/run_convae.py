import getpass
user_id = getpass.getuser()

import numpy as np
import pandas as pd
from patient_reps import *


def gen_codes(num_samples, num_features, input_file):  
    low_code = 0
    high_code = 100  
    codes = np.random.randint(low_code, high_code, size=(num_samples, num_features))

    ids = np.array(range(1, num_samples+1))

    num_samples = 100
    num_features = 1000
    ids, codes = gen_codes(num_samples, num_features)
    output_file = os.path.join(data_path, input_file)
    with open(output_file, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "ENCODED-AVG"])
        for m, e in zip(ids, codes):
            wr.writerow([m] + list(e))

    return ids, codes


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

data_path = f"/home/{user_id}/OTTEHR/embedding/convae_architecture/data"

# Generate input file
input_file = 'code.csv'
num_samples = 100
num_features = 1000
# gen_codes(num_samples, num_features, input_file)


# Generate vocabulary and save to data_path/vocab.csv
# gen_vocab(list(range(low_code, high_code+1)), data_path)


# run learn_patient_representations
seed = 2
np.random.seed(seed)
output_file = f"code_emb_{seed}.csv"
learn_patient_representations(data_path, input_file, output_file, seed=seed)





