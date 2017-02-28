-- require 'path'
require 'lfs'
require 'nn'

--local BatchCreator = torch.class('util.BatchCreator') 
BatchCreator = {}
BatchCreator.__index = BatchCreator
function BatchCreator.create(input_params)
	local self = {}
	setmetatable(self, BatchCreator)
	self.inputfile = input_params.input_file
    self.vocabfile = input_params.vocabFile
    self.tensorfile = input_params.tensorFile
	self.train_frac = input_params.train_frac
    self.test_frac = input_params.test_frac
	self.valid_frac = math.max(input_params.valid_frac,1 - self.train_frac - self.test_frac)
    self.data_dir = input_params.data_dir
    return self
end

function BatchCreator:checkLoadedVectors()
    local data_dir = self.data_dir
    print(self.data_dir)
	input_file = path.join(self.data_dir,self.inputfile )
	vocab_file = path.join(self.data_dir, "vocab.t7")
	tensor_file = path.join(self.data_dir, "data.t7")
    self.input_file = input_file
    self.vocab_file = vocab_file
    self.tensor_file = tensor_file
	if not (path.exists(vocab_file) and path.exists(tensor_file)) then
		print("Have to preprocess!")
		self:createVocab()
		return 1
	else 
        if (path.exists(vocab_file)) then
		    print("Have to build tensors")
		    BatchCreator.createTensor()
		    return 1
        end
	end	
	input_attr = lfs.attributes(input_file)
	vocab_attr = lfs.attributes(vocab_file)
	tensor_attr = lfs.attributes(tensor_file)
	if input_attr.modification > vocab_attr.modification or vocab_attr.modification > tensor_attr.modification then
		createVocab()
		return 1	
	end
end
-- ** static method ** --
function binary(number, size)
    s = torch.Tensor(size)
    this_number = number
    for i=1,size do
        if(this_number%2 > 0) then
            s[1] = 1
        end
        this_number = math.floor(this.number/2)
    end
    return s
end

function BatchCreator:createTensor()
    token_count = self.token_count
    vector_list = {}
    vector_mapping = {}
    self.logsize = 1 + math.floor(math.log(#token_count))
    for k,v in pairs(token_count) do
        vector_list[k] = binary(1, self.logsize)
        vector_mapping[vector_list[k]] = v
    end
    self.vector_list = vector_list
    self.vector_mapping = vector_mapping
    vector = {}
    vect.vector_list = vector_list
    vect.vector_mapping = vector_mapping
    torch.save(self.tensor_file,vect)
end

function BatchCreator:createVocab()
    print(self)
	self:tokenize()
    token_list = self.token_list
    token_count = {}
    for i=1,#token_list do
        if token_count[token_list[i]] ~= nil then
            token_count[token_list[i]] = token_count[token_list[i]] + 1
        else
            token_count[token_list[i]] = 1
        end
    end
    token_count["UNK"] = 1
    for k,v in pairs(token_count) do
        if v < self.min_count then
            token_count[k] = nil
            token_count["UNK"] = token_count["UNK"] + v
        end
    end
    self.token_list = token_list
    torch.save(self.vocab_file,token_list)
    createTensor()
end

function BatchCreator:tokenize()
	local rawdata
    token_list = {}
	local tot_len = 0
    local escape_chars = {";",",","!","'","\""}
    local seperate_stuff = {" : ","http://"}
    local abbrev = {"Mr.", "Mrs.", "Dr."}
    for _,i in ipairs(abbrev) do
        abbrev[i] = true
    end
	local f = assert(io.open(self.input_file, "r"))
    while(1) do
		rawdata = f:read("*line")
		s = rawdata
		if not rawdata then break end
		for i=1,#escape_chars do
			s,_ = string.gsub(s,escape_chars[i]," . ")
		end
		for i=1,#seperate_stuff do
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
                if i:match("^[%w.]+@[%w+%.]*%w+$") or i:match("^[%w+%.]*%.%w+$") then 
                    add = true
                end 
            else
                if count ~= 0 then
                    if i:match("^[a-zA-Z0-9]*$") then
                        add = true
                    end
                    if abbrev[i] then
                        add = true
                    end
                end
            end
            if add then
                if token_list[i] ~= nil then
                    token_list[i] = token_list[i] + 1
                else
                    token_list[i] = 1
                end
            end
        end
	end
    self.token_list = token_list
end

function BatchCreator:createBatch()
    skip_window = self.skip_window or 2
    vector_list = self.vector_list
    linedata = self.linedata
    linevectors = {}
    for i=1,#linedata do
        s = {}
        for j=1,skip_window do 
            s[#s + 1] = vector_list["UNK"]
        end
        for j=1,#linedata[i] do
            s[#s+1] = vector_size[linedata[i][j]]
        end
        for j=1,skip_window do 
            s[#s + 1] = vector_list["UNK"]
        end
        linevectors[#linevectors + 1] = s
    end
    self.batches = linevectors
end

function BatchCreator:divideBatches()
    local batchLines = self.batches
    local trainNum = math.floor(self.train_frac*#batchLines)
    local validNum = math.floor(self.valid_frac*#batchLines)
    local testNum = #batchLines - trainNum - validNum
    self.trainBatches = subrange(batchLines,1,trainNum)
    self.validNum = subrange(batchLines,trainNum+1,trainNum+validNum)
    self.testNum = subrange(batchLines, trainNum+validNum+1,#batchLines)
end

function BatchCreator:nextBatch(batch_size, num_lines, batchType,self)
    local lines
    if batchType == 1 then lines = self.trainBatches else 
        if batchType == 2 then 
            lines = self.validBatches
        else 
            lines = self.testBatches
        end
    end
    local choosen_lines = {}
    batch = {}
    for i=1,num_lines do
        rand = math.floor(math.random()*(#lines))
        choosen_lines[#choosen_lines + 1] = lines[rand]
    end
    for i=1,#choosen_lines do
        if #batch > batch_size then break end
        for j=skip_window+1,#choosen_lines[i] - skip_window do
            batchvectors = {}
            for k=1,skip_window do
                batchvectors[#batchvectors + 1] = choosen_lines[i-k]
                batchvectors[#batchvectors + 1] = choosen_lines[i+k]
            end
            batch[choosen_lines[i][j]] = batchvectors
        end
    end
    batch = subrange(batch,1,batch_size)
    return batch
end

return BatchCreator
