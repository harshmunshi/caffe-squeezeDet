#encoding=utf-8
# ---------------------------------
# caffe-squeezeDet
# Written by XuSenhai
# ---------------------------------

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import caffe
import numpy as np
import os
import pdb

absroot = os.getcwd()

def main():
    """
    This function transform npy into caffemodel
    """

    # Set gpu model
    caffe.set_mode_gpu()

    # Set net file path
    net_file = os.path.join(absroot, 'prototxt', \
                            'squeezedet.prototxt')
    
    # Set npz model
    npz_file = os.path.join(absroot, 'model_checkpoints', \
                            'intermediate_squeezeDet.npy')

    # Initial net model
    net = caffe.Net(net_file, caffe.TRAIN)
    
    # Load npz model
    #npz_model = dict(np.load(npz_file))
    npz_model = np.load(npz_file).item()
    print(npz_model)
    
    # Set caffemodel savepath
    savepath = os.path.join(absroot, 'model_checkpoints', \
                            'final_squeezeDet.caffemodel')

    paramnames = net.params.keys()
    print(paramnames)

    for paramname in paramnames:
        npy_weights = npz_model[paramname]['weights']
        npy_biases = npz_model[paramname]['biases']
     
        net.params[paramname][0].data[...] = npy_weights
        net.params[paramname][1].data[...] = npy_biases
        
        print('Saving {} weights, biases'.format(paramname))

    net.save(savepath)
    print('Save caffemodel done!')

if __name__ == '__main__':
    main()
