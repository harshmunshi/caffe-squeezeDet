# Modified tensorpack demo code for this application

import argparse
import numpy as np
import os
import six
import tensorflow as tf

from tensorpack.tfutils import varmanip
from tensorpack.tfutils.common import get_op_tensor_name

def make_b_w_dict(list_of_layers):
    blankdict = {}
    for i in range(len(list_of_layers)):
        blankdict[list_of_layers[i]] = {}
        blankdict[list_of_layers[i]]["weights"] = 0
        blankdict[list_of_layers[i]]["biases"] = 0
    return blankdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep only TRAINABLE and MODEL variables in a checkpoint.')
    parser.add_argument('--meta', help='metagraph file', required=True)
    parser.add_argument(dest='input', help='input model file, has to be a TF checkpoint')
    parser.add_argument(dest='output', help='output model file, can be npz or TF checkpoint')
    args = parser.parse_args()



    all_layers = ['conv1', 'fire2/squeeze1x1', 'fire2/expand1x1', 'fire2/expand3x3', 'fire3/squeeze1x1', 'fire3/expand1x1', 'fire3/expand3x3', 'fire4/squeeze1x1', 'fire4/expand1x1', 'fire4/expand3x3', 'fire5/squeeze1x1', 'fire5/expand1x1', 'fire5/expand3x3', 'fire6/squeeze1x1', 'fire6/expand1x1', 'fire6/expand3x3', 'fire7/squeeze1x1', 'fire7/expand1x1', 'fire7/expand3x3', 'fire8/squeeze1x1', 'fire8/expand1x1', 'fire8/expand3x3', 'fire9/squeeze1x1', 'fire9/expand1x1', 'fire9/expand3x3', 'fire10/squeeze1x1', 'fire10/expand1x1', 'fire10/expand3x3', 'fire11/squeeze1x1', 'fire11/expand1x1', 'fire11/expand3x3', 'conv12']
    # this script does not need GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    final_dict = make_b_w_dict(all_layers)
    print(final_dict)

    try:
        tf.train.import_meta_graph(args.meta, clear_devices=True)
    except KeyError:
        print("If your graph contains non-standard ops, you need to import the relevant library first.")
        raise

    # loading...
    if args.input.endswith('.npz'):
        dic = np.load(args.input)
    else:
        dic = varmanip.load_chkpt_vars(args.input)
    dic = {get_op_tensor_name(k)[1]: v for k, v in six.iteritems(dic)}
    #print(dic)

    for layer in all_layers:
        final_dict[layer]["weights"] = dic[layer+"/kernels:0"].T
        final_dict[layer]["biases"] = dic[layer+"/biases:0"]

    print(final_dict)
    np.save("intermediate_squeezeDet.npy", final_dict)