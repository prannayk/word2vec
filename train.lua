require 'nn'
require 'lfs'

input_params = {
    data_dir = "~/machinelearning/datasets/",
    learningRate = 0.1,
    dropout = 0.2,
    skip_window = 2,
    margin = 0.02,
    num_layers = 3,
    filename = "text8.txt",
    train_frac = 0.7,
    valid_frac = 0.2,
    test_frac = 0.1,
    input_file = "text8.txt",
    vocab_file = "vocab.t7",
    tensor_file = "tensor.t7"
}

print("Loaded dependencies")

local BatchCreator = require 'util.BatchCreator'(input_params)
BatchCreator:checkLoaded()

print ("Loaded data, finished building Vocab and Tensors")
