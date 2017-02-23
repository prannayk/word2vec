require 'torch'
require 'nn'

local hiererchial_model = {}
function hiererchial_model.create_net(input_size, dropout,vocab_size)
    skip_window = self.skip_window or 2
    vector_size = 0
    logrange = math.floor(math.log(vocab_size))
    sizerange = logrange**3 - range^3 - logrange^2
    if(self.vector_Size ~= nil) then
        vector_size = self.vector_size
    else
        vector_size = math.random()*sizerange + logrange^2
    end
    output_size = skipwindow*2*vector_size
    dropout = dropout or 0
    numlayers = self.numlayers or numlayers or 3
    net = nn.Sequential()
    net:add(nn.Linear(input_size,output_size))
    for i=1.num_layers do
        net:add(nn.Linear(output_size,output_size))
    end
    net:add(nn.Sigmoid())
    self.net = net
end
