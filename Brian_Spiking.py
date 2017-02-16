import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
import scipy
from struct import unpack
import cPickle as pickle


from brian2 import *
# specify the location of the input data
Brian_data_path = ''


#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
def get_labeled_data(picklename, bTrain=True):
    """Read input-vector (image) and target class (label,0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(Brian_data_path + 'train-images.idx3-ubyte', 'rb')
            labels = open(Brian_data_path + 'train-labels.idx1-ubyte', 'rb')
        else:
            images = open(Brian_data_path + 't10k-images.idx3-ubyte', 'rb')
            labels = open(Brian_data_path + 't10k-labels.idx1-ubyte', 'rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in xrange(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)] for unused_row in xrange(rows)]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print readout.shape, fileName
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr

#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------
start_scope()
start = time.time()
training = get_labeled_data(Brian_data_path + 'training')
end = time.time()
print 'time needed to load training set:', end - start

start = time.time()
testing = get_labeled_data(Brian_data_path + 'testing', bTrain = False)
end = time.time()
print 'time needed to load test set:', end - start

#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = False


np.random.seed(0)
data_path = './'





ending = ''
n_input = 784
n_e = 64
n_i = n_e

tau = 10*ms

eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''
G = NeuronGroup(3, eqs, threshold='v>1', reset='v = 0', method='linear')
G.I = [2, 0, 0]
G.tau = [10, 100, 100]*ms

# Comment these two lines out to see what happens without Synapses
S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
S.connect(i=0, j=[1, 2])
S.w = 'j*0.2'

M = StateMonitor(G, 'v', record=True)

run(50*ms)

plt.plot(M.t/ms, M.v[0], '-b', label='Neuron 0')
plt.plot(M.t/ms, M.v[1], '-g', lw=2, label='Neuron 1')
plt.plot(M.t/ms, M.v[2], '-r', lw=2, label='Neuron 1')
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.legend(loc='best')
plt.show()