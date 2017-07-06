"""Hidden Markov Model sequence tagger

"""
from classifier import Classifier
import numpy
import itertools

class HMM(Classifier):
		
	def get_model(self): return None
	def set_model(self, model): pass

	model = property(get_model, set_model)
		
	def _collect_counts(self, instance_list):
		"""Collect counts necessary for fitting parameters

		This function should update self.transtion_count_table
		and self.feature_count_table based on this new given instance

		Add your docstring here explaining how you implement this function

		Returns None
		"""	
		index = 1
		lab_index = 1
		for instance in instance_list:
			curr = -1
			last = -1
			for feat, label in itertools.izip(instance.data, instance.label):
			
				if feat not in self.w2i:
					self.w2i[feat] = index
					self.i2w[index] = feat
					index += 1
					#increasing the size of the count table
					self.feature_count_table = numpy.concatenate((self.feature_count_table, numpy.zeros((long(self.feature_count_table.shape[0]), 1))), 1)
				if label not in self.label2index:
					self.label2index[label] = lab_index
					self.index2label[lab_index] = label
					lab_index += 1
					#increasing the size of the count tables
					if(lab_index != 1):
						self.feature_count_table = numpy.concatenate((self.feature_count_table, numpy.zeros((1, long(self.feature_count_table.shape[1])))), 0)
						#expanding the size of the transition tables in both dimensions
						self.transition_count_table = numpy.concatenate((self.transition_count_table, numpy.zeros((long(self.transition_count_table.shape[0]), 1))), 1)
						self.transition_count_table = numpy.concatenate((self.transition_count_table, numpy.zeros((1, long(self.transition_count_table.shape[1])))), 0)
						#expanding the size of the termination count
						self.termination_count = numpy.concatenate((self.termination_count, numpy.zeros((1, long(self.termination_count.shape[1])))), 0)
						
				#incrementing the count for label and feature
				self.feature_count_table[self.label2index[label], self.w2i[feat]] += 1
				self.feature_count_table[self.label2index[label], 0] += 1
				
				last = curr
				curr = self.label2index[label]
				
				if last != -1:
					self.transition_count_table[last, curr] += 1
				else:
					self.transition_count_table[curr, 0] += 1
					
			self.termination_count[curr, 0] += 1
		
		#finished counting
		# print(self.feature_count_table)
		# print(self.transition_count_table)
		
		pass

	def train(self, instance_list):
		"""Fit parameters for hidden markov model

		Update codebooks from the given data to be consistent with
		the probability tables 

		Transition matrix and emission probability matrix
		will then be populated with the maximum likelihood estimate 
		of the appropriate parameters

		Add your docstring here explaining how you implement this function

		Returns None
		"""
		self.transition_matrix = numpy.zeros((1,1))
		self.emission_matrix = numpy.zeros((1,1))
		self.transition_count_table = numpy.zeros((1,1))
		self.feature_count_table = numpy.zeros((1,1))
		self.termination_count = numpy.zeros((1,1))
		self.termination_matrix = numpy.zeros((1,1))
		self.w2i = {}
		self.i2w = {}
		self.label2index = {}
		self.index2label = {}
		self._collect_counts(instance_list)
		#TODO: estimate the parameters from the count tables
		#smoothing constant
		smoothing = 1
		
		self.transition_matrix = numpy.zeros((self.transition_count_table.shape[0], self.transition_count_table.shape[1]))
		self.emission_matrix = numpy.zeros((self.feature_count_table.shape[0], self.feature_count_table.shape[1]))
		self.termination_matrix = numpy.zeros((self.termination_count.shape[0], self.termination_count.shape[1]))
		
		#summing the total number of initial states
		initsum = 0
		for i in range(1, self.transition_matrix.shape[0]):
			initsum += self.transition_count_table[i,0]
		
		for i in range(1, self.transition_matrix.shape[0]):
			#obtaining the initial starting hidden state probabilities
			self.transition_matrix[i, 0] = float(self.transition_count_table[i,0]+ smoothing) / float(initsum + (smoothing*len(self.label2index)))
			
			#finding the probabilities for transition matrix for state to state
			total = 0
			for j in range(1, self.transition_matrix.shape[1]):
				total += self.transition_count_table[i, j]
			for j in range(1, self.transition_matrix.shape[1]):
				self.transition_matrix[i,j] = float(self.transition_count_table[i,j] + smoothing) / float(total + (smoothing*len(self.label2index)))
		
		#populating the emission matrix
		for i in range(1, self.emission_matrix.shape[0]):
			for j in range(self.emission_matrix.shape[1]):
				self.emission_matrix[i,j] = float(self.feature_count_table[i,j] + smoothing) / float(self.feature_count_table[i,0] + (smoothing*len(self.w2i)))
				
		#populating the termination matrix
		for i in range(1, self.termination_count.shape[0]):
			self.termination_matrix[i, 0] = float(self.termination_count[i, 0] + smoothing) / float(initsum + (smoothing*len(self.label2index)))
		
		return None

	def classify(self, instance):
		"""Viterbi decoding algorithm

		Wrapper for running the Viterbi algorithm
		We can then obtain the best sequence of labels from the backtrace pointers matrix

		Add your docstring here explaining how you implement this function

		Returns a list of labels e.g. ['B','I','O','O','B']
		"""
		backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)
		index = backtrace_pointers[backtrace_pointers.shape[0]-1, backtrace_pointers.shape[1]-1]
		best_sequence = []
		best_sequence.append(index)
		for i in range(backtrace_pointers.shape[1]-1, 0, -1):
			best_sequence.insert(0, backtrace_pointers[index, i])
			index = backtrace_pointers[index, i]
		
		for i in range(0, len(best_sequence)):
			best_sequence[i] = self.index2label[best_sequence[i]]
			
		return best_sequence

	def compute_observation_loglikelihood(self, instance):
		"""Compute and return log P(X|parameters) = loglikelihood of observations"""
		trellis = self.dynamic_programming_on_trellis(instance, True)
		loglikelihood = trellis[trellis.shape[0]-1, trellis.shape[1]-1]
		return loglikelihood

	def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
		"""Run Forward algorithm or Viterbi algorithm

		This function uses the trellis to implement dynamic
		programming algorithm for obtaining the best sequence
		of labels given the observations

		Add your docstring here explaining how you implement this function

		Returns trellis filled up with the forward probabilities 
		and backtrace pointers for finding the best sequence
		"""
		#TODO:Initialize trellis and backtrace pointers 
		trellis = numpy.zeros((self.transition_matrix.shape[0] + 1, len(instance.data)))
		backtrace_pointers = numpy.zeros((trellis.shape[0],trellis.shape[1]))
		#TODO:Traverse through the trellis here
		
		if run_forward_alg:
			#initial step
			for i in range(1, trellis.shape[0]-1):
				# in case the feature was never seen before
				trellis[i,0] = self.transition_matrix[i,0]
				if instance.data[0] in self.w2i:
					trellis[i,0] = trellis[i,0]*self.emission_matrix[i, self.w2i[instance.data[0]]]
				
			#recursive step
			for o in range(1, trellis.shape[1]):
				for q in range (1, trellis.shape[0]-1):
					for qpast in range (1, trellis.shape[0]-1):
						# summing up all of possibilities of the possible transitions to this state
						trellis[q,o] += trellis[qpast, o-1]*self.transition_matrix[qpast, q]
					if instance.data[o] in self.w2i:
						trellis[q,o] = trellis[q,o]*self.emission_matrix[q, self.w2i[instance.data[o]]]
			
			#termination step?
			for q in range(1, trellis.shape[0]-1):
				trellis[trellis.shape[0], trellis.shape[1]] += trellis[q, trellis.shape[1]]*self.termination_matrix[q, 0]
			
			return trellis
		else:
			#initial step
			for i in range(1, trellis.shape[0]-1):
				trellis[i,0] = self.transition_matrix[i,0]
				# in case the feature was never seen before
				if instance.data[0] in self.w2i:
					trellis[i,0] = trellis[i,0]*self.emission_matrix[i, self.w2i[instance.data[0]]]
					
			#recursive step
			for o in range(1, trellis.shape[1]):
				for q in range (1, trellis.shape[0]-1):
					for qpast in range (1, trellis.shape[0]-1):
						# updating to the most likely transition
						if trellis[qpast, o-1]*self.transition_matrix[qpast, q] > trellis[q,o]:
							trellis[q,o] = trellis[qpast, o-1]*self.transition_matrix[qpast, q]
							backtrace_pointers[q,o] = qpast
					if instance.data[o] in self.w2i:
						trellis[q,o] = trellis[q,o]*self.emission_matrix[q, self.w2i[instance.data[o]]]
						
			#termination step?
			for q in range(1, trellis.shape[0]-1):
				if trellis[q, trellis.shape[1]-1] > trellis[trellis.shape[0]-1, trellis.shape[1]-1]:
					trellis[trellis.shape[0]-1, trellis.shape[1]-1] = trellis[q, trellis.shape[1]-1]*self.termination_matrix[q, 0]
					backtrace_pointers[backtrace_pointers.shape[0]-1, backtrace_pointers.shape[1]-1] = q
			
			return backtrace_pointers

	def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
		"""Baum-Welch algorithm for fitting HMM from unlabeled data (EXTRA CREDIT)

		The algorithm first initializes the model with the labeled data if given.
		The model is initialized randomly otherwise. Then it runs 
		Baum-Welch algorithm to enhance the model with more data.

		Add your docstring here explaining how you implement this function

		Returns None
		"""
		if labeled_instance_list is not None:
			self.train(labeled_instance_list)
		else:
			#TODO: initialize the model randomly
			pass
		while True:
			#E-Step
			self.expected_transition_counts = numpy.zeros((1,1))
			self.expected_feature_counts = numpy.zeros((1,1))
			for instance in instance_list:
				(alpha_table, beta_table) = self._run_forward_backward(instance)
				#TODO: update the expected count tables based on alphas and betas
				#also combine the expected count with the observed counts from the labeled data
			#M-Step
			#TODO: reestimate the parameters
			if self._has_converged(old_likelihood, likelihood):
				break

	def _has_converged(self, old_likelihood, likelihood):
		"""Determine whether the parameters have converged or not (EXTRA CREDIT)

		Returns True if the parameters have converged.	
		"""
		return True

	def _run_forward_backward(self, instance):
		"""Forward-backward algorithm for HMM using trellis (EXTRA CREDIT)

		Fill up the alpha and beta trellises (the same notation as 
		presented in the lecture and Martin and Jurafsky)
		You can reuse your forward algorithm here

		return a tuple of tables consisting of alpha and beta tables
		"""
		alpha_table = numpy.zeros((1,1))
		beta_table = numpy.zeros((1,1))
		#TODO: implement forward backward algorithm right here

		return (alpha_table, beta_table)

