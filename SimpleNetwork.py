import tensorflow as tf

class outputFunctions():
	def __init__(self, inputs, weights, biases):
		self.inputs = inputs
		self.weights = weights
		self.biases = biases
		self.output = None
	
	def softmax(self):
		self.output = tf.nn.softmax(tf.matmul(self.inputs,self.weights)+self.biases)

class setNetwork():

	def __init__(self, inputDim, groundTruthDimensions):
		self.placeholderXdim = inputDim
		self.placeholderYdim = groundTruthDimensions
		self.tfxPlace = None
		self.tfyPlace = None
		self.weights = None
		self.biases = None
		self.output = None

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

	def setup(self, optimizer):
		self.dim2placeholder()
		self.setWeights()
		self.setBiases()
		self.output = outputFunctions(self.tfxPlace, self.weights, self.biases)
		print self.output.biases
		outstring =  'self.output.%s()' %optimizer
		self.output = eval(outstring)




net = setNetwork([None, 784],[None, 10])
net.setup('softmax')