import numpy as np
import tensorflow as tf
rnn_cell=tf.nn.rnn_cell
#rnn_cell=tf.contrib.rnn.core_rnn_cell #run in the cpu-only computer





class LSTM_CNN():
	def __init__(self,batch_size,lstm_size,lstm_layer,dim_image,
		input_embedding_size,vocabulary_size,drop_out_rate,dim_hidden,max_words_q,output_size,dim_img_emd):

		self.batch_size=batch_size
		self.lstm_size=lstm_size
		self.lstm_layer=lstm_layer
		self.dim_image=dim_image
		self.input_embedding_size=input_embedding_size
		self.vocabulary_size=vocabulary_size
		self.drop_out_rate=drop_out_rate
		self.dim_hidden=dim_hidden
		self.max_words_q=max_words_q
		self.output_size=output_size 
		self.dim_img_emd=dim_img_emd
	#variables
		#question feature
		self.ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
		self.embed_ques_W = tf.Variable(tf.random_uniform([vocabulary_size, input_embedding_size], -0.08, 0.08), name='embed_ques_W')
		self.lstm_1 = rnn_cell.LSTMCell(lstm_size,  use_peepholes=True, state_is_tuple=True)
		self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = 1 - drop_out_rate)
		self.lstm_2 = rnn_cell.LSTMCell(lstm_size , use_peepholes=True, state_is_tuple=True)
		self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = 1 - drop_out_rate)
		self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2], state_is_tuple=True)
		#self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_1, self.lstm_2], state_is_tuple=True)
		self.init_state=self.stacked_lstm.zero_state(self.batch_size, tf.float32)
		

		#attention layer
		self.embed_state_W = tf.Variable(tf.random_uniform([2*lstm_size*lstm_layer, dim_hidden], -0.08,0.08),name='embed_state_W')
		self.embed_state_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_state_b')
		#image embedding
		self.embed_image_W = tf.Variable(tf.random_uniform([self.dim_img_emd, self.dim_hidden], -0.08, 0.08), name='embed_image_W')
		self.embed_image_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_image_b')
		self.embed_scor_W = tf.Variable(tf.random_uniform([self.dim_hidden,self.output_size], -0.08, 0.08), name='embed_scor_W')
		self.embed_scor_b = tf.Variable(tf.random_uniform([self.output_size], -0.08, 0.08), name='embed_scor_b')

		#self.score_weight=tf.Variable(tf.random_uniform([att_dim,output_size], -0.08,0.08), name='score_weight')
		#self.score_bias = tf.Variable(tf.random_uniform([output_size], -0.08, 0.08), name='score_bias')


	def build_model(self):
		if type(self.dim_image) is list:
			image_p = tf.placeholder(tf.float32, [self.batch_size]+self.dim_image)
			#image = tf.squeeze(image_p)
			image= tf.reshape(image_p,[self.batch_size,self.dim_img_emd])
		else:
			image_p = tf.placeholder(tf.float32, [self.batch_size,self.dim_image])
			image=image_p
		question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
		label = tf.placeholder(tf.int64, [self.batch_size,])

		state = self.init_state
		loss = 0.0
		with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
			for i in range(self.max_words_q):
				if i>0:
					tf.get_variable_scope().reuse_variables()
					self.ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:,i-1])
				ques_emb_drop = tf.nn.dropout(self.ques_emb_linear, 1-self.drop_out_rate)
				#ques_emb_drop = self.ques_emb_linear
				ques_emb = tf.tanh(ques_emb_drop)
				output, state = self.stacked_lstm(ques_emb, state)
			image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)	
		#image_drop= image
		
		state0,state1=tf.split(value=state, num_or_size_splits=2,axis=0) 
		state00, state01=tf.split(value=state0, num_or_size_splits=2,axis=1)
		state10, state11=tf.split(value=state1, num_or_size_splits=2,axis=1)
		'''
	        state0,state1=tf.split(value=state, num_split=2,split_dim=0) 
        	state00, state01=tf.split(value=state0, num_split=2,split_dim=1)
        	state10, state11=tf.split(value=state1, num_split=2,split_dim=1)
		'''
		#state=tf.concat(1,[tf.squeeze(state00),tf.squeeze(state01),tf.squeeze(state10),tf.squeeze(state11)])
		state=tf.concat([tf.squeeze(state00),tf.squeeze(state01),tf.squeeze(state10),tf.squeeze(state11)],0)
		state = tf.expand_dims(state, 0)
		state_drop = tf.nn.dropout(state, 1-self.drop_out_rate)
		#state_drop=state
		state_linear = tf.nn.xw_plus_b(state_drop, self.embed_state_W, self.embed_state_b)
		state_emb = tf.tanh(state_linear)
		#state_emb_drop = tf.nn.dropout(state_emb, 1-self.drop_out_rate)
		#image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
		image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
		image_emb = tf.tanh(image_linear)
		#image_emb_drop = tf.nn.dropout(image_emb, 1-self.drop_out_rate)
		scores = tf.multiply(state_emb, image_emb)
		#scores = tf.mul(state_emb_drop, image_emb_drop)
		scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
		scores_emb = tf.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b) 
		prob_ans= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_emb, labels=label) #original (no dropout)
		#prob_ans= tf.nn.sparse_softmax_cross_entropy_with_logits(score_drop, label) 
		loss = tf.reduce_mean(prob_ans)

		return loss, image_p, question, label

	def build_generator(self):
		if type(self.dim_image) is list:
			image_p = tf.placeholder(tf.float32, [self.batch_size]+self.dim_image)
			#image = tf.squeeze(image_p)
			image= tf.reshape(image_p,[self.batch_size,self.dim_img_emd])
		else:
			image_p = tf.placeholder(tf.float32, [self.batch_size,self.dim_image])
			image=image_p
		question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
		label = tf.placeholder(tf.int64, [self.batch_size,])

		state = self.init_state
		loss = 0.0
		for i in range(self.max_words_q):
			if i>0:
				tf.get_variable_scope().reuse_variables()
				self.ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:,i-1])
			ques_emb_drop = tf.nn.dropout(self.ques_emb_linear, 1-self.drop_out_rate)
			#ques_emb_drop = self.ques_emb_linear
			ques_emb = tf.tanh(ques_emb_drop)
			output, state = self.stacked_lstm(ques_emb, state)
		image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)	
		#image_drop= image
		'''
		state0,state1=tf.split(value=state, num_split=2,split_dim=0) 
		state00, state01=tf.split(value=state0, num_split=2,split_dim=1)
		state10, state11=tf.split(value=state1, num_split=2,split_dim=1)
		'''
		state0,state1=tf.split(value=state, num_or_size_splits=2,axis=0)
	        state00, state01=tf.split(value=state0, num_or_size_splits=2,axis=1)
        	state10, state11=tf.split(value=state1, num_or_size_splits=2, axis=1)
		state=tf.concat([tf.squeeze(state00),tf.squeeze(state01),tf.squeeze(state10),tf.squeeze(state11)],0)
		state = tf.expand_dims(state, 0)
		#state=tf.concat(1,[tf.squeeze(state00),tf.squeeze(state01),tf.squeeze(state10),tf.squeeze(state11)])
		#state=tf.concat(1,[tf.squeeze(state00,[1,2]),tf.squeeze(state01,[1,2]),tf.squeeze(state10,[1,2]),tf.squeeze(state11,[1,2])])
		state_drop = tf.nn.dropout(state, 1-self.drop_out_rate)
		#state_drop=state
		state_linear = tf.nn.xw_plus_b(state_drop, self.embed_state_W, self.embed_state_b)
		state_emb = tf.tanh(state_linear)
		#state_emb_drop = tf.nn.dropout(state_emb, 1-self.drop_out_rate)
		#image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
		image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
		image_emb = tf.tanh(image_linear)
		#image_emb_drop = tf.nn.dropout(image_emb, 1-self.drop_out_rate)
		scores = tf.multiply(state_emb, image_emb)
		#scores = tf.mul(state_emb_drop, image_emb_drop)
		scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
		scores_emb = tf.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b)
		
		

		return scores_emb, image_p, question
