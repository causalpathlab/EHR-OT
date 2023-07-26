import os
import sys
sys.path.append("/home/wanxinli/EHR-OT/EHR-OT")
from da import DA
# from queue import queue
import queue as queue
from multiprocessing import Queue
import timeit


class SDA(object):

    """
    Stacked Denoising Autoencoder (SDA)
    """

    def __init__(self, nvisible, nhidden=50, nlayer=3, param={}):
        self.nlayer = nlayer
        print('initializing: %d-layer SDAs\n' % self.nlayer)

        corrupt_lvl = Queue()
        self._tune_corruption_level(corrupt_lvl, param)

        self.sda = []
        for i in range(1, self.nlayer + 1):
            param['corrupt_lvl'] = corrupt_lvl.get()

            if i == 1:
                da = DA(nvisible, nhidden, param, i)
            else:
                da = DA(nhidden, nhidden, param, i)

            self.sda.append(da)
        return


    def train(self, data):
        """
        Train the stack of denoising autoencoders from data

        @param data: matrix samples x features
        """
        dt = data

        print('training: %d-layer SDAs\n' % self.nlayer)

        start_time = timeit.default_timer()

        for i in range(self.nlayer):

            self.sda[i].train(dt)

            if i < self.nlayer - 1:
                print('applying: DA [layer: %d]\n' % self.sda[i].layer)
                dt = self.sda[i].apply(dt)

        end_time = timeit.default_timer()
        print('\ntraining time: %.2f sec.\n' % (end_time - start_time))
        return


    def apply(self, data):
        """
        Apply the stack of denoising autoencoders to data

        @param data: matrix samples x features
        """
        print('applying: %d-layer SDA' % self.nlayer)

        dt = data

        for i in range(self.nlayer):
            print('(*) applying: DA [layer: %d]' % self.sda[i].layer)
            dt = self.sda[i].apply(dt)

        return dt
    
    def reconstructed_input(self, h):
        """
        Compute the reconstrcuted data given the hidden representation
        """

        dt = h

        for i in range(self.nlayer):
            print('(*) applying: DA [layer: %d]' % self.sda[i].layer)
            dt = self.sda[i].reconstructed_input(dt)
        
        return dt


    # private functions

    def _tune_corruption_level(self, q, param):
        """
        Set corruption level
        """
        try:
            v = param['corrupt_lvl']
        except Exception:
            v = 0.01

        c = v
        for i in range(self.nlayer):
            q.put(c)
        return
