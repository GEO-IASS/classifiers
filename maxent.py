# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from math import exp
import numpy as np
import random

class MaxEnt(Classifier):

	def get_model(self): return None

	def set_model(self, model): pass

	model = property(get_model, set_model)
	MXmodel = {}
	previters = []

	def train(self, instances, dev_instances=None):
		"""Construct a statistical model from labeled instances."""
		self.train_sgd(instances, dev_instances, .5, 70)

	def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
		#initializing the model to zeros
		self.MXmodel  = {}
		self.previters = []
		self.MXmodel["weights"] = np.zeros((1,1))
		self.MXmodel["gradient"] = np.zeros((1,1))
		
		#initializing the convergence condition
		converged = 0
		#integer to keep track of iteration number
		iternum = 0;
		while converged == 0:
			
			#create new batch, collecting data on if we need to expand weights matrix
			batch = []
			largestfeat = self.MXmodel["gradient"].shape[0]
			largestlab = self.MXmodel["gradient"].shape[1]
			for x in range(batch_size):
				randinst = train_instances.__getitem__(random.randrange(0,train_instances.__len__()))
				for feature in randinst.data:
					if feature+1 > largestfeat:
						largestfeat = feature+1	#keeping track of if the gradient/weight matrix needs to be expanded
				if randinst.label+1 > largestlab:
					largestlab = randinst.label+1
				batch.append(randinst)
			
			#expand matrix if its too small
			if long(largestfeat) > self.MXmodel["gradient"].shape[0]:
				self.MXmodel["gradient"] = np.concatenate((self.MXmodel["gradient"], np.zeros((long(largestfeat)-self.MXmodel["gradient"].shape[0],self.MXmodel["gradient"].shape[1]))), 0)
				self.MXmodel["weights"] = np.concatenate((self.MXmodel["weights"], np.zeros((long(largestfeat)-self.MXmodel["weights"].shape[0],self.MXmodel["weights"].shape[1]))), 0)
			
			if long(largestlab) > self.MXmodel["gradient"].shape[1]:
				self.MXmodel["gradient"] = np.concatenate((self.MXmodel["gradient"], np.zeros((self.MXmodel["gradient"].shape[0], long(largestlab)-self.MXmodel["gradient"].shape[1]))), 1)
				self.MXmodel["weights"] = np.concatenate((self.MXmodel["weights"], np.zeros((self.MXmodel["weights"].shape[0], long(largestlab)-self.MXmodel["weights"].shape[1]))), 1)
			
			# calculate gradient
			# calculate observed
			observed = np.zeros(self.MXmodel["gradient"].shape)
			for instance in batch:
				lab = instance.label
				feats = instance.data
				for feat in feats:
					observed[feat,lab] += 1
			
			#calculate expected
			expected = np.zeros(self.MXmodel["gradient"].shape)
			
			for instance in batch:
				#because the denominator of the posterior should be the same for all classifications, we can calculate that first to save some calculation time
				#denominator of posterior
				denom = 0.
				for x in range(0, self.MXmodel["weights"].shape[1]):
					sum = 0.
					for feat in instance.data:
						sum += self.MXmodel["weights"][feat,x]
					denom += exp(sum)
				
				#calculating the numerator of each posterior for each classification
				for x in range(0, self.MXmodel["gradient"].shape[1]):
					numer = 0.
					for feat in instance.data:
						numer += self.MXmodel["weights"][feat,x]
					numer = exp(numer)
					
					intergrad = numer/denom
					
					#adjusting the expected matrix for all of the features that show up for this calculation
					for feat in instance.data:
						expected[feat,x] = expected[feat,x] + intergrad
			
			#calculate the gradient using observed and expected values
			self.MXmodel["gradient"] = np.subtract(observed, expected)
			
			#update the weights depending on the gradient and the learning rate
			self.MXmodel["weights"] = np.add(self.MXmodel["weights"], np.multiply(learning_rate, self.MXmodel["gradient"]))

			#checking the accuracy of the score model weights on the dev set
			correct = 0.
			total = 0.
			for inst in dev_instances:
				if self.classify(inst) == inst.label:
					correct += 1.
				total += 1.
			
			#calculating the accuracy
			score = correct/total
			
			#decaying the learning rate, but making sure it remains above a certain threshold
			if learning_rate > .001:
				learning_rate = learning_rate * .97
			
			#printing the iteration number used to see progress of training
			print(iternum, correct, total, score)
			
			#checking convergence. If this and all of the past 9 iterations have a lower score (accuracy) than the 10th past iteration,
			#it the model is set to be the 10th past iteration
			if len(self.previters) < 10:
				self.previters.append((score, self.MXmodel["weights"], iternum))
			else:
				self.previters.pop(0)
				self.previters.append((score, self.MXmodel["weights"], iternum))
				timetostop = 1
				
				#checking to see if any of the last 9 iterations have a higher score than the 10th last iteration
				for ind in range(1, len(self.previters)):
					if self.previters[0][0] < self.previters[ind][0]:
						timetostop = 0
				
				#if none of the last 9 were as good, set weights to be that of the best iteration
				if timetostop == 1:
					converged = 1
					print("chosen:", self.previters[0][2], "score: " ,self.previters[0][0] )
					self.MXmodel["weights"] = self.previters[0][1]
			
			#incrementing the iteration number
			iternum += 1
			
			#shorten the "better than last x" requirement if runs for too long to make sure it doesnt run infinitely
			if iternum%30 == 0:
				self.previters.pop(0)
			
		# print(self.MXmodel["weights"])
		pass

	def classify(self, instance):
		for feat in instance.data:
			#so it wont crash if there was somehow a classification in the dev set not yet seen
			if feat >= self.MXmodel["weights"].shape[0]:
				return 0
	
		#the index of the highest classification, as well as the value of it
		highestclass = 0
		highestval = 0
		for x in range(0,self.MXmodel["weights"].shape[1]):
		
			#calculate the probability for each classification
			weightsum = 0
			for feat in instance.data:
				weightsum += self.MXmodel["weights"][feat,x]
			
			#updates the highest classification score (the one that will be output) if the current classification is higher than any previous
			if exp(weightsum) > highestval:
				highestval = exp(weightsum)
				highestclass = x
		
		return highestclass

    
