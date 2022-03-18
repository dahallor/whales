from queue import Full
from activations import ReLuLayer, SigmoidLayer, SoftmaxLayer, TanhLayer
from layers import *
import numpy as np
import random
import getopt
import sys
import math
import matplotlib.pyplot as plt
from cnn import ConvolutionalLayer

# seed to 0 for reproducability
random.seed(0)

inputData = np.array([1,2,3,4])

LEARNING_RATE = 0.01
VAL_PERCENT = 1/3
MAX_EPOCHS = 10000
BATCH_SIZE = 665

class MulticlassNetwork:
	def __init__(self, trainpath, valpath, middle_layers, objective, val_percent=VAL_PERCENT, learning_rate=LEARNING_RATE, max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
		self.learning_rate = learning_rate
		self.max_epochs = max_epochs
		self.batch_size = batch_size

		self.middle_layers = middle_layers
		self.objective = objective

		self.setup(trainpath, valpath, val_percent)

	def one_hot_encode(self, predictions):
		n_values = np.max(predictions) + 1
		return np.eye(int(n_values))[predictions]

	def setup(self, trainpath, valpath, val_percent):
		# val_percent is a float in range [0,1] indicating how much of the data to allocate to training vs validation
		train_set = []
		train_pred = []
		val_set = []
		val_pred = []

		with open(trainpath, "r") as dataset:
			for row in dataset:
				row = row.split(",")
				convertedRow = [float(x) for x in row]
				train_set.append(convertedRow[1:len(convertedRow)])
				train_pred.append(convertedRow[0])

		with open(valpath, "r") as dataset:
			for row in dataset:
				row = row.split(",")
				convertedRow = [float(x) for x in row]
				val_set.append(convertedRow[1:len(convertedRow)])
				val_pred.append(convertedRow[0])

		shuffled_train = random.sample(range(len(train_set)), len(train_set))
		shuffled_val = random.sample(range(len(train_set)), len(train_set))

		self.train_set = np.array([train_set[x] for x in shuffled_train])
		self.train_pred = self.one_hot_encode([int(train_pred[x]) for x in shuffled_train])

		self.val_set = np.array([val_set[x] for x in shuffled_val])
		self.val_pred = self.one_hot_encode([int(val_pred[x]) for x in shuffled_val])

		self.input_layer = InputLayer(self.train_set)

	def learn(self):
		self.epoch = 0
		epochs = []
		train_accs = []
		val_accs = []

		while (self.epoch < self.max_epochs):
			acc, obj = self.train_epoch(self.train_set, self.train_pred, self.batch_size)
			val_acc, val_obj = self.validate_epoch(self.val_set, self.val_pred)

			if self.epoch % 100 == 0:
				print(f"Completed epoch {self.epoch}. Train Acc: {acc}. Val Acc: {val_acc}")
			self.epoch += 1

			acc, obj = self.train_epoch(self.train_set, self.train_pred, self.batch_size)
			val_acc, val_obj = self.validate_epoch(self.val_set, self.val_pred)

			train_accs.append(obj)
			val_accs.append(val_obj)
			epochs.append(self.epoch)

			if epoch % 100 == 0:
				print(f"Completed epoch {epoch}. Train Acc: {acc}. Val Acc: {val_acc}")
			epoch += 1

		plt.xlabel("Epochs")
		plt.ylabel("Log Loss")
		plt.plot(epochs, train_accs, label = 'Training Objective', color = "black")
		plt.plot(epochs, val_accs, label = 'Validation Objective', color = "black")
		plt.legend()
		plt.show()

		print(f"Training Accuracy: {acc}, Validation Accuracy: {val_acc}")

	def get_multiclass_accuracy(self, y, yhat):
		return np.sum(np.argmax(yhat, axis=1) == np.argmax(y, axis=1)) / len(y)

	def forward_prop(self, train_set, train_pred):
		input = self.input_layer.forward(train_set)
		for layer in self.middle_layers:
			input = layer.forward(input)
		return self.objective.eval(train_pred, input), input
		
	def backprop(self, train_pred, obj):
		grad = self.objective.gradient(train_pred, obj)
		for layer in reversed(self.middle_layers):
			grad = layer.backward(grad, LEARNING_RATE)

	def minibatch(self, train_set, train_pred, batchsize):
		indices = np.arange(train_set.shape[0])
		np.random.shuffle(indices)
		for start_idx in range(0, train_set.shape[0] - batchsize + 1, batchsize):
			excerpt = indices[start_idx:start_idx + batchsize]
			yield train_set[excerpt], train_pred[excerpt]

	def train_epoch(self, train_set, train_pred, batch_size):
		acc_list = []
		objs = []
		for batch in self.minibatch(train_set, train_pred, batch_size):
			curr_train, curr_pred = batch
			obj, yhat = self.forward_prop(curr_train, curr_pred)
			self.backprop(curr_pred, obj)
			acc_list.append(self.get_multiclass_accuracy(curr_pred, yhat))
			objs.append(obj)
		return np.mean(acc_list), np.mean(objs)

	def validate_epoch(self, val_set, val_pred):
		obj, yhat = self.forward_prop(val_set, val_pred)
		accuracy = self.get_multiclass_accuracy(val_pred, yhat)
		return accuracy, obj

def testANN1():
	middle_layers = [
		FullyConnectedLayer(0, 10),
		SigmoidLayer(),
		FullyConnectedLayer(10, 10),
		SigmoidLayer(),
		FullyConnectedLayer(10, 0),
		SigmoidLayer(),
		]

	objective = LogLoss()


	network = MulticlassNetwork("mnist_train_100.csv", "mnist_valid_10.csv", middle_layers, objective)
	network.learn()


def testWhaleCNN():
	middle_layers = [
		ConvolutionalLayer(),
		SigmoidLayer(),
		FullyConnectedLayer(10, 10),
		SigmoidLayer(),
		FullyConnectedLayer(10, 0),
		SigmoidLayer(),
		]

	objective = LogLoss()


	network = MulticlassNetwork("mnist_train_100.csv", "mnist_valid_10.csv", middle_layers, objective)
	network.learn()

def main(argv):
    # This parameter parsing used the follwing page for reference:
    # https://www.tutorialspoint.com/python/python_command_line_arguments.htm
    opts, args = getopt.getopt(argv,"t:n:", ["tlr", "ann1", "ann2", "ann3", "mlp1", "mlp2", "mlp3"])
    for opt, arg in opts:
        if opt == '--cnn':
            testWhaleCNN()

if __name__ == "__main__":
	main(sys.argv[1:])