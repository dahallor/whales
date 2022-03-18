from queue import Full
from drop_out_layer import DropOutLayer
from layers import *
from cnn import ConvolutionalLayer
from activations import *
from main import MulticlassNetwork
from pool import *

LEARNING_RATE = 0.00001
MAX_EPOCHS = 100
BATCH_SIZE = 10

class MulticlassNetwork:
	def __init__(self, train_set, objective, learning_rate=LEARNING_RATE, max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
		self.learning_rate = learning_rate
		self.max_epochs = max_epochs
		self.batch_size = batch_size

		self.middle_layers = middle_layers
		self.objective = objective

		self.setup(train_set)

	def one_hot_encode(self, predictions):
		n_values = np.max(predictions) + 1
		return np.eye(int(n_values))[predictions]

	def setup(self, train_set):
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

def testWhaleCNN(test_set):
	middle_layers = [
	ConvolutionalLayer()
	SigmoidLayer()
	PoolingLayer()
	ConvolutionalLayer()
	SigmoidLayer()
	FullyConnectedLayer(9, 5)
	DropOutLayer()
	FullyConnectedLayer(5,5)
	SoftmaxLayer()
		]

	objective = CrossEntropy()

	network = MulticlassNetwork(middle_layers, objective)
	
	network.learn()
