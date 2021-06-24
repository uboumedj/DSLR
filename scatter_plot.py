import sys
import pandas as pd
import numpy as np
import math
import argparse
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
import matplotlib.pyplot as plt
from describe import read_source_data, trim_and_treat_data

def plot_figure(dataset, numerical_dataset, class_1, class_2):
	"""
	Decides between plotting all the scatter graphs for every class, or only the
	user-requested classes - if there are.

	Parameters:
	dataset (pandas.DataFrame): the original data structure obtained from the source file
	numerical_dataset (pandas.DataFrame): the numerical data extracted from the dataset
	class_1, class_2 (string): the user-requested classes
	"""
	if (class_1 == "all" or class_2 == "all"):
		plot_all_scatter_plots(dataset, numerical_dataset)
	else:
		try:
			plot_selected_scatter_plot(dataset, numerical_dataset, class_1, class_2)
		except:
			print("\033[1;91mError: \033[0mOne of the requested classes doesn't exist!")
			sys.exit(1)

def plot_all_scatter_plots(dataset, numerical_dataset):
	"""
	Plots and displays all the possible scatter plots (i.e for each pair of classes) in
	a single window

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
			fill_scatter_plot(ax, dataset, class_1, class_2, 0.1)
			i += 1	
	if (len(numerical_dataset.columns)):
		fig.tight_layout()
		plt.show()
	else:
		print("\033[1;91mError: \033[0mNo data to display in a pair plot!")

def plot_selected_scatter_plot(dataset, numerical_dataset, class_1, class_2):
	"""
	Plots a single scatter plot graph showing the correlation of grades between two classes

	Parameters:
	dataset (pandas.DataFrame): the original data structure obtained from the source file
	numerical_dataset (pandas.DataFrame): the numerical data extracted from the dataset
	class_1, class_2 (string): the user-requested classes
	"""
	plt.style.use('classic')
	fig = plt.figure()
	fig.suptitle("Correlation between " + class_1 + " and " + class_2)
	ax = fig.add_subplot(1, 1, 1)
	ax.set_xlabel(class_1)
	ax.set_ylabel(class_2)
	fill_scatter_plot(ax, dataset, class_1, class_2, 0.2)
	plt.show()

def fill_scatter_plot(ax, dataset, class_1, class_2, transparency):
	"""
	Fills the scatter plot graph with the markers for the specified pairing of variables

	Parameters:
	ax (matplotlib.axes): the current graph's object
	dataset (pandas.DataFrame): the original data structure obtained from the source file
	class_1, class_2 (string): the chosen classes
	transparency: the transparency of the graph's markers
	"""
	houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
	colors = ["#B92100", "#e8c227", "#6693fc", "#008700"]
	for house, color in zip(houses, colors):
		ax.scatter(dataset[dataset["Hogwarts House"] == house][class_1], dataset[dataset["Hogwarts House"] == house][class_2], color=color, alpha=transparency)
	if (dataset['Hogwarts House'].isnull().sum() > 0):
		ax.scatter(dataset[class_1], dataset[class_2], alpha=0.7, color='black')

def parse_arguments():
	"""
	Sets up the argument parser's various parameters

	Returns:
	(Namespace): a dictionary returned by argparse, with the program's parameters associated to their value 
	"""
	parser = argparse.ArgumentParser(description="Creates standard scatter plots for the given dataset")
	parser.add_argument('-d', '--data', type=str, help="CSV file containing the dataset", default="./datasets/dataset_train.csv")
	parser.add_argument('-x', '--x_axis', type=str, help="a class name, among Hogwarts' many classes (default displays all)", default="all")
	parser.add_argument('-y', '--y_axis', type=str, help="another class to compare the first one to", default="all")
	return(parser.parse_args())

def main(argv):
	args = parse_arguments()
	dataset = read_source_data(args.data)
	numerical_dataset = trim_and_treat_data(dataset, "numeric", False)
	plot_figure(dataset, numerical_dataset, args.x_axis, args.y_axis)

if __name__ == "__main__":
	main(sys.argv)