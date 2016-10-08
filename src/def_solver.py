import sys
import os

caffe_root = '../../../caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2

def create_solver(base_learning_rate, regularization_param, train_net_path, test_net_path):
	
	s = caffe_pb2.SolverParameter()

	# Set a seed for reproducible experiments:
	# this controls for randomization in training.
	s.random_seed = 0xCAFFE

	# Specify locations of the train and (maybe) test networks.
	s.train_net = train_net_path

	s.test_net.append(test_net_path)
	s.test_iter.append(1) # Test on 100 batches each time we test.

	s.test_net.append(train_net_path)
	s.test_iter.append(1) # Test on 100 batches each time we test.


	s.test_interval = 1000000  # Test after every 500 training iterations.


	s.max_iter = 10000     # no. of times to update the net (training iterations)
	 
	# EDIT HERE to try different solvers
	# solver types include "SGD", "Adam", and "Nesterov" among others.
	s.type = "Adam"

	# Set the initial learning rate for SGD.
	s.base_lr = base_learning_rate  # EDIT HERE to try different learning rates
	# Set momentum to accelerate learning by
	# taking weighted average of current and previous updates.
	s.momentum = 0.9
	# Set weight decay to regularize and prevent overfitting
	s.weight_decay = regularization_param

	# Set `lr_policy` to define how the learning rate changes during training.
	# This is the same policy as our default LeNet.
	s.lr_policy = 'inv'
	s.gamma = 0.0001
	s.power = 0.75
	# EDIT HERE to try the fixed rate (and compare with adaptive solvers)
	# `fixed` is the simplest policy that keeps the learning rate constant.
	# s.lr_policy = 'fixed'

	# Display the current training loss and accuracy every 1000 iterations.
	s.display = 2

	# Snapshots are files used to store networks we've trained.
	# We'll snapshot every 5K iterations -- twice during training.
	s.snapshot = 1000000
	s.snapshot_prefix = 'lenet_'

	s.snapshot_after_train = True

	# Train on the GPU
	s.solver_mode = caffe_pb2.SolverParameter.CPU

	return s