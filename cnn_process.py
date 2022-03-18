from queue import Full
from drop_out_layer import DropOutLayer
from layers import *
from cnn import ConvolutionalLayer
from activations import *
from main import MulticlassNetwork
from pool import *

def setup():
	# train_percent is a float in range [0,1] indicating how much of the data to allocate to training vs validation
	TRAIN_PERCENT = 2/3

	observations = []
	predictions = []
	skip = True
	with open("KidCreative.csv", "r") as dataset:
		# Skip header row
		for row in dataset:
			if skip:
				skip = False
				continue
			row = row.split(",")
			convertedRow = [float(x) for x in row]
			observations.append(convertedRow[2:len(convertedRow)])
			predictions.append(convertedRow[1])

	train_num = math.ceil(len(observations) * TRAIN_PERCENT)

	shuffled_indices = random.sample(range(len(observations)), len(observations))

	self.train_set = np.array([observations[x] for x in shuffled_indices[0:train_num-1]])
	self.train_pred = np.array([predictions[x] for x in shuffled_indices[0:train_num-1]])

	self.val_set = np.array([observations[x] for x in shuffled_indices[train_num-1:len(observations)]])
	self.val_pred = np.array([predictions[x] for x in shuffled_indices[train_num-1:len(observations)]])

	self.MIN_MAPE = 10**-10
	self.MAX_EPOCHS = 10000
	self.LEARNING_RATE = 10**-4

	self.input = InputLayer(self.train_set)
	self.hidden1 = FullyConnectedLayer(16,1)
	self.activation = SigmoidLayer()
	self.objective = LogLoss()

	self.epoch = 0
	self.train_ll = []
	self.val_ll = []

def learn(self):

	mape = math.inf

	while (self.epoch < self.MAX_EPOCHS and mape > self.MIN_MAPE):
		ll, acc = self.train_epoch()
		self.train_ll.append(ll)
		val_ll, val_acc = self.validate_epoch()
		self.val_ll.append(val_ll)

		self.epoch += 1

	plt.plot(range(self.epoch), self.train_ll, label="Training")
	plt.plot(range(self.epoch), self.val_ll, label="Validation")
	plt.xlabel("Epochs")
	plt.ylabel("Log Loss")
	plt.legend()
	plt.show()

	print(f"Training Accuracy: {acc}, Validation Accuracy: {val_acc}")

def train_epoch(self):
	l1 = self.input.forward(self.train_set)
	l2 = self.hidden1.forward(l1)
	l3 = self.activation.forward(l2)
	l4 = self.objective.eval(self.train_pred, l3)

	g1 = self.objective.gradient(self.train_pred, l3) #TODO check
	g2 = self.activation.backward(g1)
	g3 = self.hidden1.backward(g2, self.LEARNING_RATE)

	correct = 0
	for i in range(len(l3)):
		pred = 0
		if l3[i] < 0.5:
			pred = 1
		else:
			pred = 0

		if pred == self.train_pred[i]:
			correct+=1

	accuracy = correct / len(l3)
	
	return np.mean(l4), accuracy

def testWhaleCNN(images, targets):
	input = InputLayer(images)
	conv1 = ConvolutionalLayer()
	activ1 = SigmoidLayer()
	pool1 = PoolingLayer()
	conv2 = ConvolutionalLayer()
	activ2 = SigmoidLayer()
	d1 = FullyConnectedLayer(9, 5)
	drop1 = DropOutLayer()
	d2 = FullyConnectedLayer(5,5)
	activ3 = SoftmaxLayer()

	objective = CrossEntropy()




	network.learn()
