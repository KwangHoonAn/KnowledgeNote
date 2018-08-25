import numpy as np
import os
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))

data = np.loadtxt(dir_path + '/data_classification.csv', delimiter=',')
featDim = len(data[0])


label = np.copy(data.T[-1])
data[:, -1] = 1



def activation(x):
	# This modell will return a probability score between 0 and 1
	return 1./(1 + np.exp(-x))
# Threshold p >= 0.5, class 1 / p < 0.5, class 0

def inference(x, w):
	linear_combination = np.dot(x, w)
	return activation(linear_combination)

#def decision():

def cost_function(y_, x_, w):
	cost = 0.0
	m = len(y_)
	for y, x in zip(y_, x_):
		hypothesis = inference(x, w)

		cost += (y*np.log(hypothesis) + (1-y)*np.log(1-hypothesis))
		# print(y, y*np.log(hypothesis), (1-y)*np.log(1-hypothesis))
	cost *= -1
	return cost/m


def main():
	epoch = 100
	learning_rate = 0.1
	# np.random.seed(0)
	weights = np.random.rand(featDim)
	weights = weights / sum(weights)
	weights_l1 = np.copy(weights)
	weights_l2 = np.copy(weights)

	hyper = 0.00001
	cost_ = []
	for i in range(epoch):
		for batch, y in zip(data, label):

			weights = weights - learning_rate*(inference(batch, weights) - y)*batch
			weights_l1 = weights_l1 - learning_rate*(inference(batch, weights_l1) - y)*batch + hyper*sum(abs(weights_l1))
			weights_l2 = weights_l2 - learning_rate*(inference(batch, weights_l2) - y)*batch + hyper*sum(np.square(weights_l2))
		cost_.append( cost_function(label, data, weights) )
	# print(weights, weights_l1, weights_l2)

	acc = 0.0
	for batch, y in zip(data, label):
		prob = inference(batch, weights)
		if y:
			if prob >= 0.5:
				acc += 1
		else:
			if prob < 0.5:
				acc += 1
	print(acc)
	plt.plot(cost_)
	plt.ylabel('Cost per epoch')
	plt.xlabel('Epoch')
	plt.show()
main()
# derivative sigmoid = sig(z)*(1-sig(z)), 

