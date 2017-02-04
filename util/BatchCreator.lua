require 'path'
require 'lfs'
require 'torch'

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
	if not (path.exists(vocab_file) and path.exists(tensor_file)) then
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

function BatchCreator:createVocab()
	token_set = tokenize()
end

function BatchCreater:tokenize()
	local rawdata
    local token_list
	local tot_len = 0
    local escape_chars = {";",",","!","'","\""}
    local seperate_stuff = {" : ","http://"}
	local f = assert(io.open(self.file, "r"))
	while(1) do
		rawdata = f:read("*line")
		s = rawdata
		if not rawdata then break end
		for i=1,#escape_chars do
			s,_ = string.gsub(s,escape_chars[i]," . ")
		end
		for i=1.#seperate_stuff do
			s,_ = string.gsub(s,seperate_stuff[i]," "..seperate_stuff[i].." ")
		end
        for i in s:gmatch("%S+") do 
            local count = 0
            for t in i:gmatch("%w+") do
                count = count + 1 
                if count > 1 then
                    break
                end
            end
            local add = false
            if count > 1 then
                if abbrev[i]==true then
                    add = true
                else
                    if i:match("^[%w.]+@%w+.%.[%a%d]+$") or i:match("^[%w.]*%[%a%d]+$") then 
                        add = true
                    end 
                end
            else
                add = true
            end
            if add then
                if token_list[i] != nil then
                    token_list[i] = token_list + 1
                else
                    token_list[i] = 1
                end
            end
        end
	end
    self.token_list = token_list
end
