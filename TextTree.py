def slice_cache(cache,start=0,end=-1):
    cache_type = type(cache)
    layers = []
    for layer in cache:
        key_tensor = layer[0][:,:,start:end,:]
        value_tensor = layer[1][:,:,start:end,:]
        layers.append((key_tensor,value_tensor))
    return cache_type(layers,model.config)





class TextTree:
	def __init__(self,parent = None,value=None,past_key_values=None,attentions = None,logits = None, tokenizer=None,model=None):
		
		self.value = value #the token textual representation
		self.past_key_values = past_key_values #the pkv so far (same as parent's except a single token is appended)
		self.attentions = attentions 
		self.logits = logits
		self.children = []
		
		self.parent = parent #dont neccasarily need this, but could be useful
		if self.parent:
			self.tokenizer = self.parent.model
			self.model = self.parent.model
		else:
			self.tokenizer = tokenizer
			self.model = model






	def add_list_of_tokens(self,tokens,outputs):
		if len(tokens) > 0:
			length_of_this_pkv = self.past_key_values[0][0].shape[2]
			child_past_key_values = slice_cache(outputs.past_key_values,0,length_of_this_pkv+1) 

			self.attentions

			#what attention to give child, that is, what attentionwould the next node see
			#making sure its square




			child_token = tokens[0]
			self.children.append(TextTree(self,child_token,child_past_key_values,child_attentions))
			if len(tokens) > 1:
				self.children[-1].add_list_of_tokens(tokens[1:],remaining_outputs)









	def get_item_by_index(self,index):
		if len(index) > 1:
			return self.children[index[0]][index[1:]]
		elif len(index) > 0:
			return self.children[index[0]]


	def add_text(self,text):
		input_ids = self.tokenizer(text,return_tensors = "pt",add_special_tokens = False)["input_ids"]
		tokens = self.tokenizer.tokenize(text)
		new_pkv = type(self.past_key_values)([layer for layer in past_key_values]) #construct new one so self.past_key_values DOES NOT GET CHANGED
		outputs = self.model(input_ids = input_ids,past_key_values=new_pkv,use_cache = True,output_attentions=True)

		self.add_list_of_tokens(tokens,outputs)



	# def get_child_by_value(self,value):
	# 	#what if two children have same value
	# 	#that should not be allowed it causes difficulty in traversing by value
	# 	#it shold also not be neccasary,
	# 	for child in self.children:
	# 		if child.value == value: return child
	# 	return None

	# def get_node_from_list_of_tokens(self,tokens_input):
	# 	assert self.value is None
	# 	child = self.get_child_by_value(tokens_input[0])
	# 	if child is None:
	# 		return None, 0
		
	# 	return child._get_node_from_list_of_tokens(tokens_input)

	# def _get_node_from_list_of_tokens(self,tokens_input):
	# 	#returns node, and a positive value for how many tokens ahead the tokens_input is from the node
	# 	token = tokens_input[0]
	# 	if self.value == token:
	# 		if len(tokens_input) == 1:
	# 			return self , 0
	# 		child = self.get_child_by_value(tokens_input[1])
	# 		if child is None:
	# 			return self , len(tokens_input) - 1
	# 		else:
	# 			return child._get_node_from_list_of_tokens(tokens_input[1:])
	# 	else:
	# 		return None, len(tokens_input)


	#TODO implement attention lookahead. tokens later in sequence, how much they attend to this one.



	#could sum the grouped queries

	#i really think the p-v2 tuning could be awesome
	#could do p-v1 just for starters
	#do it with the GLOW like generative model.
	#the prior being the objective function of the task (which could be minimizing perplexity over the desired response)
	#could even condition on a prompt, so it would essentially distill a prompt into a single token prefix  (can do multi-token prefix however)