import torch
import os

# dataset filenames
dt_files = {'ehr-file': 'syn-female-ehrseq.csv',
            # 'ehr-file-test': 'cohort_test-ehrseq.csv',
            'vocab': 'vocab.csv'}

# model parameters
model_param = {'num_epochs': 5,
               'batch_size': 32,
               'embedding_size': 5,
               'kernel_size': 2,
               'learning_rate': 0.0001,
               'weight_decay': 1e-5
               }

# length of padded sub-sequences
# len_padded = 10
len_padded = 100 # no need to pad for mimic

# save the best model
def save_best_model(epoch, model, optimizer, loss, outdir):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, os.path.join(outdir, 'best_model.pt'))
