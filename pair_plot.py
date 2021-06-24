import sys
import pandas as pd
import numpy as np
import math
import argparse
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
import matplotlib.pyplot as plt
from describe import read_source_data, trim_and_treat_data
from histogram import fill_histogram
from scatter_plot import fill_scatter_plot

def pair_plot(dataset, numerical_dataset):
	"""
	Creates and displays the pair plot, which consists in a scatter plot for each variable pairing,
	and a histogram for the plots along the diagonal (when the pairing would be between the same variable)

	Parameters:
	dataset (pandas.DataFrame): the original data structure obtained from the source file
	numerical_dataset (pandas.DataFrame): the numerical data extracted from the dataset
	"""
	plt.style.use('classic')
	plt.rcParams.update({'font.size': 9})
	fig = plt.figure(figsize=(25,15))
	i = 1
	size = len(numerical_dataset.columns)
	for class_1 in numerical_dataset.columns:
		for class_2 in numerical_dataset.columns:
			ax = fig.add_subplot(size, size, i)
			if (i <= size):
				ax.set_title(class_2)
			if (i % size == 1):
				ax.set_ylabel(class_1[0:6])
			ax.set_yticklabels([])
			ax.set_xticklabels([])
			if (class_1 != class_2):
				fill_scatter_plot(ax, dataset, class_1, class_2, 0.1)
			else:
				fill_histogram(ax, dataset, class_1)
			i += 1
	if (len(numerical_dataset.columns)):
		fig.tight_layout()
		plt.show()
	else:
		print("\033[1;91mError: \033[0mNo data to display in a pair plot!")

def parse_arguments():
	"""
	Sets up the argument parser's various parameters

	Returns:
	(Namespace): a dictionary returned by argparse, with the program's parameters associated to their value 
	"""
	parser = argparse.ArgumentParser(description="Creates a pair plot for the given dataset")
	parser.add_argument('-d', '--data', type=str, help="CSV file containing the dataset", default="./datasets/dataset_train.csv")
	return(parser.parse_args())

def main(argv):
	args = parse_arguments()
	dataset = read_source_data(args.data)
	numerical_dataset = trim_and_treat_data(dataset, "numeric", False)
	pair_plot(dataset, numerical_dataset)

if __name__ == "__main__":
	main(sys.argv)