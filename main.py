##
## Copyright (C) IBM Corp, 2018
## Copyright (C) Huan Zhang <huan@huan-zhang.com>, 2018
## Copyright (C) Tsui-Wei Weng  <twweng@mit.edu>, 2018
##
## This program is licenced under the Apache-2.0 licence,
## contained in the LICENCE file in this directory.
## 

# ignore warning that the numpy throws due to numpy
import warnings
warnings.filterwarnings("ignore")

import save_nlayer_weights as nl
import numpy as np

import argparse

from setup_mnist import MNIST
from setup_cifar import CIFAR

import tensorflow as tf
import os
import sys
import random
import time
import logging

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.autograph.set_verbosity(0)

def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)

# from setup_imagenet import ImageNet, ImageNetModel
set_tf_loglevel(logging.FATAL)

# to re-enable error loggin use this
# set_tf_loglevel(logging.INFO)

from utils import generate_data
from PIL import Image

# import our bounds
from get_bounds_ours import get_weights_list, compute_worst_bound, compute_worst_bound_multi
# import others bounds: currently, LP bound is called in the compute_worst_bound
from get_bounds_others import spectral_bound

if __name__ == "__main__":
    
    #### parser ####
    parser = argparse.ArgumentParser(description='compute activation bound for CIFAR and MNIST')
    parser.add_argument('--model', 
                        default="nn_veri",
                        choices=["mnist", "cifar", "nn_veri"],
                        help='model to be used')
    parser.add_argument('--init_state',
                        default='[0, 0]',
                        help='initial state of the system. Please enter as a list e.g init state at 0,0 will be [0,0].'
                             'Note: Do not leave empty space after the comma')
    parser.add_argument('--eps',
                        default=0.1,
                        type=float,
                        help="epsilon for verification")
    parser.add_argument('--hidden',
                        default=8,
                        type=int,
                        help="number of hidden neurons per layer")
    parser.add_argument('--numlayer',
                        default=2,
                        type=int,
                        help='number of layers in the model')
    parser.add_argument('--numimage',
                        default=10,
                        type=int,
                        help='number of images to run')
    parser.add_argument('--startimage',
                        default=0,
                        type=int,
                        help='start image')
    parser.add_argument('--norm',
                        default="i",
                        type=str,
                        choices=["i", "1", "2"],
                        help='perturbation norm: "i": Linf, "1": L1, "2": L2')
    parser.add_argument('--method',
                        default="adaptive",
                        type=str,
                        choices=["general", "ours", "adaptive", "spectral", "naive"],
                        help='"ours": our proposed bound, "spectral": spectral norm bounds, "naive": naive bound')
    parser.add_argument('--lipsbnd',
                        type=str,
                        default="disable",
                        choices=["disable", "fast", "naive", "both"],
                        help='compute Lipschitz bound, after using some method to compute neuron lower/upper bounds')
    parser.add_argument('--lipsteps',
                        type=int,
                        default=30,
                        help='number of steps to use in lipschitz bound')
    parser.add_argument('--LP',
                        action="store_true",
                        help='use LP to get bounds for final output')
    parser.add_argument('--LPFULL',
                        action="store_true",
                        help='use FULL LP to get bounds for output')
    parser.add_argument('--quad',
                        action="store_true",
                        help='use quadratic bound to imporve 2nd layer output')
    parser.add_argument('--warmup',
                        default=" ",
                        action="store_true",
                        help='warm up before the first iteration')
    parser.add_argument('--modeltype',
                        default="vanilla",
                        choices=["vanilla", "dropout", "distill", "adv_retrain"],
                        help="select model type")
    parser.add_argument('--targettype',
                        default="untargeted",
                        choices=["untargeted", "least", "top2", "random"],
                        help='untargeted minimum distortion')
    parser.add_argument('--steps',
                        default=15,
                        type=int,
                        help='how many steps to binary search')
    parser.add_argument('--activation',
                        default="relu",
                        choices=["relu", "tanh", "sigmoid", "arctan", "elu", "hard_sigmoid", "softplus"])

    args = parser.parse_args()

    # parse init_state which is a string into a number
    init_state = list(map(float, args.init_state.strip('[]').split(',')))
    # init_state = list(map(float, args.init_state.strip().split()))[:2]

    nhidden = args.hidden
    # quadratic bound only works for ReLU
    assert ((not args.quad) or args.activation == "relu")
    # for all activations we can use general framework
    assert args.method == "general" or args.activation == "relu"

    targeted = True
    if args.targettype == "least":
        target_type = 0b0100
    elif args.targettype == "top2":
        target_type = 0b0001
    elif args.targettype == "random":
        target_type = 0b0010
    elif args.targettype == "untargeted":
        target_type = 0b10000
        targeted = False

    if args.modeltype == "vanilla":
        suffix = ""
    else:
        suffix = "_" + args.modeltype

    if args.model == "nn_veri":
        activation = "relu"
        path = os.path.dirname(os.path.abspath(__file__)) + "/models/my_model"
        modelfile = path
    # else:
    #     # try models/mnist_3layer_relu_1024
    #     activation = args.activation
    #     modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_" + activation + "_" + str(nhidden) + suffix
    #     if not os.path.isfile(modelfile):
    #         # if not found, try models/mnist_3layer_relu_1024_1024
    #         modelfile += ("_"+str(nhidden))*(args.numlayer-2) + suffix
    #         # if still not found, try models/mnist_3layer_relu
    #         if not os.path.isfile(modelfile):
    #             modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_" + activation + "_" + suffix
    #             # if still not found, try models/mnist_3layer_relu_1024_best
    #             if not os.path.isfile(modelfile):
    #                 modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_" + activation + "_" + str(nhidden) + suffix + "_best"
    #                 if not os.path.isfile(modelfile):
    #                     raise(RuntimeError("cannot find model file"))

    if args.LP or args.LPFULL:
        # use gurobi solver
        import gurobipy as grb

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:   
        if args.model == "mnist":
            data = MNIST()
            model = nl.NLayerModel(params=[nhidden] * (args.numlayer - 1),
                                   restore=modelfile,
                                   activation=activation,
                                   use_personal=False)
        elif args.model == "cifar":
            data = CIFAR()
            model = nl.NLayerModel(params=[nhidden] * (args.numlayer - 1),
                                   restore=modelfile,
                                   image_size=32,
                                   image_channel=3,
                                   activation=activation,
                                   use_personal=False)
        elif args.model == "nn_veri":
            # write my own function here
            model = nl.NLayerModel(params=[nhidden] * (args.numlayer - 1),
                                   restore=modelfile,
                                   use_personal=True,
                                   input_dimension=2)
        else:
            raise(RuntimeError("unknown model: "+args.model))
                
        print("Evaluating", modelfile)
        sys.stdout.flush()

        random.seed(1215)
        np.random.seed(1215)
        tf.set_random_seed(1215)

        # the weights and bias are saved in lists: weights and bias
        # weights[i-1] gives the ith layer of weight and so on
        weights, biases = get_weights_list(model)        

        if not args.model == "nn_veri":
            inputs, targets, true_labels, true_ids, img_info = generate_data(data,
                                                                             samples=args.numimage,
                                                                             targeted=targeted,
                                                                             random_and_least_likely=True,
                                                                             target_type=target_type,
                                                                             predictor=model.model.predict,
                                                                             start=args.startimage)
            # get the logit layer predictions
            preds = model.model.predict(inputs)

        Nsamp = 0
        r_sum = 0.0
        r_gx_sum = 0.0
        
        # warmup
        if args.warmup:
            print("warming up...")
            sys.stdout.flush()
            if args.method == "spectral":
                robustness_gx = spectral_bound(weights,
                                               biases,
                                               0,
                                               1,
                                               inputs[0],
                                               preds[0],
                                               args.numlayer,
                                               args.norm,
                                               not targeted)
            else:
                compute_worst_bound(weights=weights,
                                    biases=biases,
                                    pred_label=0,
                                    target_label=1,
                                    x0=np.array(init_state),
                                    predictions=0,
                                    numlayer=args.numlayer,
                                    p=args.norm,
                                    eps=args.eps,
                                    method=args.method,
                                    lipsbnd=args.lipsbnd,
                                    is_LP=args.LP,
                                    is_LPFULL=args.LPFULL,
                                    untargeted=not targeted,
                                    use_quad=args.quad,
                                    nn_veri=True)

                if args.model == "nn_veri":
                    print("*******************************Done Computing bounds*******************************")
                    sys.exit()

        # print("starting robustness verification on {} images!".format(len(inputs)))
        sys.stdout.flush()
        sys.stderr.flush()
        total_time_start = time.time()

        for i in range(len(inputs)):
            Nsamp += 1                
            p = args.norm # p = "1", "2", or "i"
            predict_label = np.argmax(true_labels[i])
            target_label = np.argmax(targets[i])
            start = time.time()
            # Spectral bound: no binary search needed
            if args.method == "spectral":
                robustness_gx = spectral_bound(weights,
                                               biases,
                                               predict_label,
                                               target_label,
                                               inputs[i],
                                               preds[i],
                                               args.numlayer,
                                               p,
                                               not targeted)
            # compute worst case bound
            # no need to pass in sess, model and data
            # just need to pass in the weights, true label, norm, x0, prediction of x0, number of layer and eps
            elif args.lipsbnd != "disable":
                # You can always use the "multi" version of Lipschitz bound to improve results (about 30%).
                robustness_gx = compute_worst_bound_multi(weights,
                                                          biases,
                                                          predict_label,
                                                          target_label,
                                                          inputs[i],
                                                          preds[i],
                                                          args.numlayer,
                                                          p,
                                                          args.eps,
                                                          args.lipsteps,
                                                          args.method,
                                                          args.lipsbnd,
                                                          not targeted)
                eps = args.eps
                # if initial eps is too small, then increase it
                if robustness_gx == eps:
                    while robustness_gx == eps:                   
                        eps = eps*2
                        print("==============================")
                        print("increase eps to {}".format(eps))
                        print("==============================")
                        robustness_gx = compute_worst_bound_multi(weights,
                                                                  biases,
                                                                  predict_label,
                                                                  target_label,
                                                                  inputs[i],
                                                                  preds[i],
                                                                  args.numlayer,
                                                                  p,
                                                                  eps,
                                                                  args.lipsteps,
                                                                  args.method,
                                                                  args.lipsbnd,
                                                                  not targeted)
                # if initial eps is too large, then decrease it
                elif robustness_gx <= eps / 5:
                    while robustness_gx <= eps / 5:
                        eps = eps/5
                        print("==============================")
                        print("increase eps to {}".format(eps))
                        print("==============================")
                        robustness_gx = compute_worst_bound_multi(weights,
                                                                  biases,
                                                                  predict_label,
                                                                  target_label,
                                                                  inputs[i],
                                                                  preds[i],
                                                                  args.numlayer,
                                                                  p,
                                                                  eps,
                                                                  args.lipsteps,
                                                                  args.method,
                                                                  args.lipsbnd,
                                                                  not targeted)
            else:
                gap_gx = 100
                eps = args.eps
                eps_LB = -1
                eps_UB = 1
                counter = 0
                is_pos = True
                is_neg = True
                
                # perform binary search
                eps_gx_UB = np.inf
                eps_gx_LB = 0.0
                # is_pos = True
                # is_neg = True
                # eps = eps_gx_LB*2
                # eps = args.eps
                while eps_gx_UB-eps_gx_LB > 0.00001:
                    gap_gx, _, _ = compute_worst_bound(weights=weights,
                                                       biases=biases,
                                                       pred_label=predict_label,
                                                       target_label=target_label,
                                                       x0=inputs[i],
                                                       predictions=preds[i],
                                                       numlayer=args.numlayer,
                                                       p=p,
                                                       eps=eps,
                                                       method=args.method,
                                                       lipsbnd="disable",
                                                       is_LP=args.LP,
                                                       is_LPFULL=args.LPFULL,
                                                       untargeted=not targeted,
                                                       use_quad=args.quad,
                                                       activation=args.activation)
                    print("[L2][binary search] step = {}, eps = {:.5f}, gap_gx = {:.2f}".format(counter, eps, gap_gx))
                    if gap_gx > 0:
                        if gap_gx < 0.01:
                            eps_gx_LB = eps
                            break    
                        if is_pos: # so far always > 0, haven't found eps_UB
                            eps_gx_LB = eps
                            eps *= 10
                        else:
                            eps_gx_LB = eps
                            eps = (eps_gx_LB + eps_gx_UB)/2
                        is_neg = False
                    else:
                        if is_neg: # so far always < 0, haven't found eps_LB
                            eps_gx_UB = eps
                            eps /= 10
                        else:
                            eps_gx_UB = eps
                            eps = (eps_gx_LB + eps_gx_UB)/2
                        is_pos = False
                    counter += 1
                    if counter >= args.steps:
                        break
                
                robustness_gx = eps_gx_LB                

            r_gx_sum += robustness_gx
            print("[L1] model = {}, seq = {}, id = {}, true_class = {}, target_class = {}, info = {}, robustness_gx = {:.5f}, avg_robustness_gx = {:.5f}, time = {:.4f}, total_time = {:.4f}".format(modelfile, i, true_ids[i], predict_label, target_label, img_info[i], robustness_gx, r_gx_sum/Nsamp, time.time() - start, time.time() - total_time_start))
            sys.stdout.flush()
            sys.stderr.flush()
            
        print("[L0] model = {}, avg robustness_gx = {:.5f}, numimage = {}, total_time = {:.4f}".format(modelfile,r_gx_sum/Nsamp,Nsamp,time.time() - total_time_start))
        sys.stdout.flush()
        sys.stderr.flush()

