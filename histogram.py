import sys
import pandas as pd
import numpy as np
import math
import argparse
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
import matplotlib.pyplot as plt
from describe import read_source_data, trim_and_treat_data

def plot_all_histograms(dataset, numerical_dataset):
	"""
	Plots the various histograms displaying the repartition of grades in each House for each class

	Parameters:
	dataset (pandas.DataFrame): the original data structure obtained from the source file
	numerical_dataset (pandas.DataFrame): the numerical data extracted from the dataset
	"""
	plt.style.use('classic')
	fig = plt.figure(figsize=(25,15))
	fig.suptitle("Grade repartition per House for each class", y=0.15)
	i = 1
	for column in numerical_dataset.columns:
		ax = fig.add_subplot(4, 4, i)
		ax.set_title(column)
		fill_histogram(ax, dataset, column)
		i += 1
	if (len(numerical_dataset.columns)):
		handles, labels = ax.get_legend_handles_labels()
		fig.tight_layout(h_pad=2.1)
		fig.legend(handles, labels, loc="lower center")
		plt.show()
	else:
		print("\033[1;91mError: \033[0mNo data to display in a histogram!")

def plot_selected_histogram(dataset, numerical_dataset, requested_class):
	"""
	Plots a histogram graph showing the repartition of grades in each House for a selected class

	Parameters:
	dataset (pandas.DataFrame): the original data structure obtained from the source file
	numerical_dataset (pandas.DataFrame): the numerical data extracted from the dataset
	requested_class (string): the user-requested class
	"""
	plt.style.use('classic')
	fig = plt.figure()
	fig.suptitle(requested_class + " Grade repartition per House")
	ax = fig.add_subplot(1, 1, 1)
	fill_histogram(ax, dataset, requested_class)
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, loc="center right")
	plt.show()

def fill_histogram(ax, dataset, requested_class):
	"""
	Fills the histogram graph with the Houses' respectively colored bars

	Parameters:
	ax (matplotlib.axes): the current graph's object
	dataset (pandas.DataFrame): the original data structure obtained from the source file
	requested_class (string): the user-requested class
	"""
	ax.hist(dataset[dataset['Hogwarts House'] == "Gryffindor"][requested_class], alpha=0.5, color='#B92100', label="Gryffindor")
	ax.hist(dataset[dataset['Hogwarts House'] == "Hufflepuff"][requested_class], alpha=0.5, color='#e8c227', label="Hufflepuff")
	ax.hist(dataset[dataset['Hogwarts House'] == "Ravenclaw"][requested_class], alpha=0.4, color='#6693fc', label="Ravenclaw")
	ax.hist(dataset[dataset['Hogwarts House'] == "Slytherin"][requested_class], alpha=0.4, color='#008700', label="Slytherin")
	if (dataset['Hogwarts House'].isnull().sum() > 0):
		ax.hist(dataset[requested_class], alpha=0.7, color='black', label="Ghost")

def plot_figure(dataset, numerical_dataset, requested_class):
	"""
	Decides between plotting all the histograms for every class, or only the
	user-requested class - if there is one.

	Parameters:
	dataset (pandas.DataFrame): the original data structure obtained from the source file
	numerical_dataset (pandas.DataFrame): the numerical data extracted from the dataset
	requested_class (string): the user-requested class
	"""
	if (requested_class == "all"):
		plot_all_histograms(dataset, numerical_dataset)
	else:
		try:
			plot_selected_histogram(dataset, numerical_dataset, requested_class)
		except:
			print("\033[1;91mError: \033[0mThe requested class doesn't exist!")
			sys.exit(1)

def parse_arguments():
	"""
	Sets up the argument parser's various parameters

	Returns:
	(Namespace): a dictionary returned by argparse, with the program's parameters associated to their value 
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data', type=str, help="CSV file containing the dataset", default="./datasets/dataset_train.csv")
	parser.add_argument('-c', '--class_name', type=str, help="a class among the various Hogwarts classes (default displays all)", default="all")
	return (parser.parse_args())

def main(argv):
	args = parse_arguments()
	dataset = read_source_data(args.data)
	numerical_dataset = trim_and_treat_data(dataset, "numeric", False)
	plot_figure(dataset, numerical_dataset, args.class_name)

if __name__ == "__main__":
	main(sys.argv)