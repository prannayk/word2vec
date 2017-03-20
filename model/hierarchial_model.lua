require 'torch'
require 'nn'

local hierarchial_model = {}
-- init function to be added

function heirarchial_model.__init__(input_params)
    self.learningRate = input_params[learningRate]
    self.skip_window = input_params[skip_window]
    self.dropout = dropout
    self.num_layers = input_params[num_layers]
    self.margin = margin
end

function hiererchial_model.create_net(input_size, dropout,vocab_size, num_layers)
    skip_window = self.skip_window or 2
    vector_size = 0
    logrange = math.floor(math.log(vocab_size))
    sizerange = logrange**3 - range^3 - logrange^2
    if(self.vector_size ~= nil) then
        vector_size = self.vector_size
    else
        vector_size = math.random()*sizerange + logrange^2
    end
    output_size = skipwindow*2*vector_size
    dropout = dropout or 0 
    dropout = dropout or self.dropout
    numlayers = self.numlayers or numlayers or 3
    net = nn.Sequential()
    net:add(nn.Linear(input_size,output_size))
    for i=1,num_layers do
        net:add(nn.Linear(output_size,output_size))
        net:add(nn.ReLU())
    end
    net:add(nn.Linear(output_size,vector_size))
    net:add(nn.Sigmoid())
    net:add(nn.Dropout(dropout, true, true, false))
    -- We have the net built up, we need the final cross entropy before we can back prop
    local margin = self.margin --  set initially 
    criterion = nn.CosineEmbeddingCriterion(margin)
    self.criterion = criterion
    self.net = net
end

function hierarchial_model.forward(input1,y,input2)
    local network_output = net:forward(input1)
    local match_output = net:forward(input2)
    local criterion_input = {}
    criterion_input[1] = network_output
    criterion_input[2] = match_output
    local errval = criterion:forward(criterion_input, y)
    net = self.net
    net:zeroGradParameters()
    local t = criterion:backward(criterion_input, y)
    net:backward(input1,t[1])
    net:backward(input2, t[2])
    net:updateParameters(self.learningRate)
end
