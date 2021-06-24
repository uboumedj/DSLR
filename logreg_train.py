import sys
import csv
import pandas as pd
import numpy as np
import argparse
import math
import time
from describe import read_source_data

class LogisticRegressionTrainer():

	def __init__(self, learning_rate=0.001, max_iterations=1500):
		"""
		Constructor for the LogisticRegressionTrainer class, initialises the various
		variables to their default value

		Parameters:
		learning_rate (float): the learning rate for the regression algorithm, which can be specified by the user with the option -l
		max_iterations (int): the maximum number of iterations for the algorithm, which can be specified with the option -i
		"""
		self.learning_rate = learning_rate
		self.max_iterations = max_iterations
		self.weights = []
		self.bias = None

	def fit(self, X, y):
		"""
		The whole logistic regression algorithm

		Parameters:
		X (numpy.ndarray): the predictor variables (the array with the students' grades)
		y (numpy.ndarray): the response variable (the array with the student's houses)

		Returns:
		(numpy.ndarray): the list of resulting weights for each house after completion of the algorithm
		"""
		X = np.insert(X, 0, 1, axis=1)
		for house in np.unique(y):
			current_house_vs_all = np.where(y == house, 1, 0)
			w = np.ones(X.shape[1])
			for _ in range(self.max_iterations):
				output = np.dot(X, w)
				errors = current_house_vs_all - self._sigmoid_function(output)
				gradient = np.dot(X.T, errors)
				w += self.learning_rate * gradient
			self.weights.append((w, house))
		return (self.weights)
    
	def predict(self, X):
		"""
		Predicts the house of each individual student

		Parameters:
		X (numpy.ndarray): every students' grades

		Returns:
		(numpy.ndarray): an array with the student's houses
		"""
		return ([self._predict_one(i) for i in np.insert(X, 0, 1, axis=1)])
    
	def compute_score(self, X, y):
		"""
		Computes the algorithm's score on the training set, i.e the percentage of right answers
		at the end of the training

		Parameters:
		X (numpy.ndarray): the predictor variables (the array with the students' grades)
		y (numpy.ndarray): the response variable (the array with the student's houses)

		Returns:
		(float): the precision score of the algorithm, between 0 and 1
		"""
		return (sum(self.predict(X) == y) / len(y))

	def _predict_one(self, grades):
		"""
		Predicts the house of one student. The function will apply the computed weights to
		the student's grades, for each House, and find out the most probable result

		Parameters:
		grades (numpy.ndarray): the students' grades

		Returns:
		(string): the student's predicted house
		"""
		max_probability = (-10, 0)
		for weight, house in self.weights:
			if ((grades.dot(weight), house) > max_probability):
				max_probability = (grades.dot(weight), house)
		return (max_probability[1])

	def _sigmoid_function(self, X):
		return (1 / (1 + np.exp(-X)))

def parse_arguments():
	"""
	Sets up the argument parser's various parameters, then parses the program's arguments and checks their values

	Returns:
	(Namespace): a dictionary returned by argparse, with the program's parameters associated to their value 
	"""
	parser = argparse.ArgumentParser(description="Simple logistic regression training")
	parser.add_argument('-d', '--data', type=str, help="CSV file containing the training dataset", default="./datasets/dataset_train.csv")
	parser.add_argument('-s', '--save', type=str, help="Name of the file where the resulting weights will be saved", default="weights")
	parser.add_argument('-i', '--iterations', type=int, help="Number of iterations for the logistic regression training", default=1500)
	parser.add_argument('-l', '--l_rate', type=float, help="Learning rate for the logistic regression training", default=0.001)
	parser.add_argument('-n', '--normalize', type=str, help="Normalization algorithm: choose between [min-max] and [z-score]", default="min-max")
	parser.add_argument('-v', '--verbose', action='store_true', help="display score and elapsed time at the end of the training")
	args = parser.parse_args()
	if (args.l_rate < 0 or args.l_rate > 0.5):
		print("\033[1;91mError: \033[0mCan't define algorithm's learning rate as negative or above 0.5")
		sys.exit(1)
	if (args.iterations < 0):
		print("\033[1;91mError: \033[0mCan't define negative iterations")
		sys.exit(1)
	if (args.normalize not in ["min-max", "z-score"]):
		print("\033[1;91mError: \033[0mNormalization algorithm doesn't exist")
		sys.exit(1)
	return (args)

def extract_data(dataset):
	"""
	Extracts the relevant columns from the original dataset and turns them into numpy.ndarrays
	for use in the logistic regression algorithm

	Returns:
	(numpy.ndarray, numpy.ndarray): respectively the column containing the target
	values for the Houses, and the columns containing class data that we will use
	"""
	if (dataset['Hogwarts House'].isnull().sum() > 0):
		print("\033[1;91mError: \033[0mCannot train with selected dataset: House data is missing!")
		sys.exit(1)
	dataset = dataset.dropna()
	try:
		target_data = np.array(dataset["Hogwarts House"])
		prediction_vars = np.array(dataset[["Herbology", "Divination", "Ancient Runes", "Charms", "Defense Against the Dark Arts"]])
	except:
		print("\033[1;91mError: \033[0mMissing critical data from dataset!")
		sys.exit(1)
	return (target_data, prediction_vars)

def pre_process_data(prediction_vars, normalize):
	"""
	Pre-processes the data by normalizing the numerical values of the class grades (applying
	either a min-max or a z-score algorithm to the elements, depending on what the user chose)

	Returns:
	(numpy.ndarray): the dataset with the new, normalized values
	"""
	if (normalize == "z-score"):
		np.apply_along_axis(z_score, 0, prediction_vars)
	else:
		np.apply_along_axis(min_max, 0, prediction_vars)
	return (prediction_vars)

def min_max(column):
	"""
	Applies a min-max normalization algorithm to a column of data, so that every value stays in
	the [0, 1] range

	Parameters:
	column (Array): a column of data in the dataset array
	"""
	mini = min(column)
	maxi = max(column)
	for index in range(len(column)):
		try:
			column[index] = ((column[index] - mini) / (maxi - mini))
		except ZeroDivisionError:
			print("\033[1;91mError: \033[0mA whole field of the dataset is equal, which makes no sense for this algorithm...")
			sys.exit(1)

def z_score(column):
	"""
	Applies a z-score normalization algorithm to a column of data, i.e every grade for a given class becomes
	the amount of standard deviations separating it from the mean

	Parameters:
	column (Array): a column of data in the dataset array
	"""
	mean = np.mean(column)
	std = np.std(column)
	for index in range(len(column)):
		try:
			column[index] = (column[index] - mean) / std
		except ZeroDivisionError:
			print("\033[1;91mError: \033[0mA whole field of the dataset is equal, which makes no sense for this algorithm...")
			sys.exit(1)

def save_result(file, weights):
	"""
	Saves the weights resulting from the logistic regression training inside a file

	Parameters:
	file (string): the name of the file
	weights (numpy.ndarray): the weights
	"""
	np.save(file, np.array(weights, dtype='object'))

def main(argv):
	args = parse_arguments()
	if (args.verbose):
		start_time = time.time()
	dataset = read_source_data(args.data)
	target_vars, prediction_vars = extract_data(dataset)
	prediction_vars = pre_process_data(prediction_vars, args.normalize)
	trainer = LogisticRegressionTrainer(learning_rate=args.l_rate, max_iterations=args.iterations)
	weights = trainer.fit(prediction_vars, target_vars)
	save_result(args.save, weights)
	if (args.verbose):
		print("Training finished in " + str(time.time() - start_time)[:5] + "s. Score obtained: " + str(trainer.compute_score(prediction_vars, target_vars))[:5])
	print("Weights saved to " + args.save + ".npy")

if __name__ == "__main__":
	main(sys.argv)