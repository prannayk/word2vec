require 'path'
require 'lfs'

local BatchCreator = torch.class('word2vec.util.BatchCreator') 

function BatchCreator:__init(data_dir, num_batches, train_frac, test_frac, valid_frac)
	local self = {}
	setmetatable(self, BatchCreator)
	self.file = filename
	self.num_batches = num_batches
	self.train_frac = train_frac
	self.test_frac = test_frac
	self.valid_frac = math.maximum(valid_frac,1 - train_frac - test_frac)
end

function BatchCreator:checkLoadedVectors(filename)
	input_file = path.join(data_dir,"input.t7" )
	vocab_file = path.join(data_dir, "vocab.t7")
	tensor_file = path.join(data_dir, "data.t7")
	if not (path.exists(vocab_file) or path.exists(tensor_file)) then
		print("Have to preprocess!")
		createVocab()
		return 1
	else (path.exists(vocab_file))
		print("Have to build tensors")
		createTensor()
		return 1
	end	
	input_attr = lfs.attributes(input_file)
	vocab_attr = lfs.attributes(vocab_file)
	tensor_attr = lfs.attributes(tensor_file)
	if input_attr.modification > vocab_attr.modification or vocab_attr.modification > tensor_attr.modification then
		createVocab()
		return 1	
	end
end

function BatchCreator:create()
	local input_attr = lfs.
end
