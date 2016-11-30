import tensorflow as tf

class outputFunctions():
	def __init__(self, inputs, weights, biases):
		self.inputs = inputs
		self.weights = weights
		self.biases = biases
		self.output = None
	
	def softmax(self):
		self.output = tf.nn.softmax(tf.matmul(self.inputs,self.weights)+self.biases)

class optimizers():
	
	def __init__(self, predictions, groundTruth):
		self.output = None
		self.groundTruth = None
		self.optimization = None
	
	def crossentropy():
		y_ = self.output
		y = self.predictionPlaceholder
		logsums = (y_ * tf.log(y))
		reduction = (-tf.reduce_sum(logsums))
		self.optimization = tf.reduce_mean(reduction, reduction_indicies =[1])

class setNetwork():

	def __init__(self, inputDim, groundTruthDimensions):
		self.placeholderXdim = inputDim
		self.placeholderYdim = groundTruthDimensions
		# data for prediction
		self.tfxPlace = None
		# input for ground truth/placeholder 
		# for predictions (gives shape for predictions)
		self.tfyPlace = None
		self.weights = None
		self.biases = None
		# prediction
		self.output = None
		self.optimizer = None

	def dim2placeholder(self):
		self.tfxPlace = tf.placeholder(tf.float32, self.placeholderXdim)
		self.tfyPlace  = tf.placeholder(tf.float32, self.placeholderYdim)

	def setWeights(self):
		x = self.placeholderXdim[1]
		y = self.placeholderYdim[1]
		self.weights = tf.Variable(tf.zeros([x,y]))

	def setBiases(self):
		y = self.placeholderYdim[1]
		self.biases = tf.Variable(tf.zeros(y))

	def setup(self, outputFunction, optimization):
		# initialization of placeholders
		setPredictionPlaceholder()
		self.dim2placeholder()
		# initialization of weights and biases
		self.setWeights()
		self.setBiases()
		# determining output function
		self.output = outputFunctions(self.tfxPlace, self.weights, self.biases)
		outstring =  'self.output.%s()' %outputFunction
		self.output = eval(outstring)
		# determining optimization function


net = setNetwork([None, 784],[None, 10])
net.setup('softmax')