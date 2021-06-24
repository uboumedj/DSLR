import sys
import csv
import pandas as pd
import numpy as np
import argparse
import math
import time
from describe import read_source_data
from logreg_train import pre_process_data

class LogisticRegressionPredictor():

	def __init__(self):
		pass

	def predict(self, X, weights):
		"""
		Predicts the house of each individual student

		Parameters:
		X (numpy.ndarray): every students' grades
		weights (numpy.ndarray): each House's weights for each class

		Returns:
		(numpy.ndarray): an array with the student's houses
		"""
		return ([self._predict_one(i, weights) for i in np.insert(X, 0, 1, axis=1)])

	def _predict_one(self, grades, weights):
		"""
		Predicts the house of one student. The function will apply the computed weights to
		the student's grades, for each House, and find out the most probable result

		Parameters:
		grades (numpy.ndarray): the students' grades
		weights (numpy.ndarray): each House's weights for each class

		Returns:
		(string): the student's predicted house
		"""
		max_probability = (-10, 0)
		for weight, house in weights:
			if ((grades.dot(weight), house) > max_probability):
				max_probability = (grades.dot(weight), house)
		return (max_probability[1])

def parse_arguments():
	"""
	Sets up the argument parser's various parameters, then parses the program's arguments and checks their values

	Returns:
	(Namespace): a dictionary returned by argparse, with the program's parameters associated to their value 
	"""
	parser = argparse.ArgumentParser(description="Prediction of a student's Hogwarts House using previously computed logistic regression model")
	parser.add_argument('-d', '--data', type=str, help="CSV file containing the test dataset", default="./datasets/dataset_test.csv")
	parser.add_argument('-w', '--weights', type=str, help="File where the computed weights are stored", default="weights.npy")
	args = parser.parse_args()
	return (args)

def get_weights(file):
	"""
	Fetches the previously stored weights in the specified file. Outputs null weights if file can't be opened

	Parameters:
	file (string): the name of the file
	"""
	try:
		weights = np.load(file, allow_pickle=True)
	except:
		print("\033[1;93mWarning: \033[0mCheck that csv file containing the weights exists and you have appropriate access rights.")
		print("Defaulting to Null weights for everything.")
		weights = []
		for house in ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]:
			weights.append(([0, 0, 0, 0, 0, 0], house))
	return (weights)

def extract_data(dataset):
	"""
	Extracts the relevant columns from the test dataset and turns them into a numpy.ndarray
	for use in the prediction algorithm

	Returns:
	(numpy.ndarray): the columns containing class data that we will use
	"""
	pd.options.mode.chained_assignment = None
	in_use_dataset = dataset[["Herbology", "Divination", "Ancient Runes", "Charms", "Defense Against the Dark Arts"]]
	for column in in_use_dataset:
		in_use_dataset[column] = in_use_dataset[column].fillna(in_use_dataset[column].mean())
	prediction_vars = np.array(in_use_dataset)
	return (prediction_vars)

def save_result(filename, houses):
	"""
	Saves the student houses resulting from the model's prediction inside a file

	Parameters:
	filename (string): the name of the file
	houses (numpy.ndarray): the student IDs associated to their houses
	"""
	with open(filename, 'w') as csvfile:
		file = csv.writer(csvfile, delimiter=',', lineterminator='\n')
		file.writerow(["Index","Hogwarts House"])
		i = 0
		for row in houses:
			file.writerow([i, row])
			i += 1
	print("Resulting houses saved in " + filename)

def main(argv):
	args = parse_arguments()
	dataset = read_source_data(args.data)
	weights = get_weights(args.weights)
	prediction_vars = extract_data(dataset)
	prediction_vars = pre_process_data(prediction_vars, "min-max")
	trainer = LogisticRegressionPredictor()
	houses = trainer.predict(prediction_vars, weights)
	save_result('houses.csv', houses)

if __name__ == "__main__":
	main(sys.argv)