import sys
import csv
import argparse
import pandas as pd
import numpy as np
import math

def parse_arguments():
	"""
	Sets up the argument parser's various parameters

	Returns:
	(Namespace): a dictionary returned by argparse, with the program's parameters associated to their value 
	"""
	parser = argparse.ArgumentParser(description="Creates a pair plot for the given dataset")
	parser.add_argument('-d', '--data', type=str, help="CSV file containing the dataset", default="./datasets/dataset_train.csv")
	parser.add_argument('-m', '--more', action='store_true', help="display more info on the dataset (non-numeric values)")
	parser.add_argument('-n', '--normalize', action='store_true', help="normalize numeric data before displaying it")
	return(parser.parse_args())

def read_source_data(file):
	"""
	Reads the source file with pandas' read_csv function and stores its content in the dataset

	Parameters:
	file (string): the name of the csv file containing the dataset

	Returns:
	(pandas.DataFrame): the dataset in a pandas.DataFrame structure
	"""
	try:
		dataset = pd.read_csv(file, index_col = "Index")
	except IOError:
		print("\033[1;91mError: \033[0mCheck that csv dataset file exists and you have appropriate access rights.")
		sys.exit(1)
	except (pd.errors.ParserError, pd.errors.EmptyDataError) as err:
		print("\033[1;91mError: \033[0mThe csv file contains corrupted data. Following is the error message from pandas:\n", err)
		sys.exit(1)
	return (dataset)

def trim_and_treat_data(dataset, type, normalize):
	"""
	Processes the data, removing unnecessary columns depending on the type passed as parameter

	Parameters:
	dataset (pandas.DataFrame): the data structure obtained from the source file
	type (string): type of data we want, either "alpha" or "numeric"
	normalize (boolean): whether the numeric data must be normalized or not

	Returns:
	(pandas.DataFrame): the trimmed dataset according to specifications
	"""
	new_dataset = dataset
	for column in new_dataset.columns:
		try:
			float(new_dataset[column][0])
			if (type == "alpha"):
				new_dataset = new_dataset.drop([column], axis=1)
		except ValueError:
			if (type == "numeric"):
				new_dataset = new_dataset.drop([column], axis=1)
			else:
				pass
	if (type == "numeric" and normalize):
		new_dataset = (new_dataset - new_dataset.min()) / (new_dataset.max() - new_dataset.min()) * 20
	new_dataset = new_dataset.dropna(axis=1, how='all')
	return (new_dataset)

def compute_standard_deviation(column, count, mean):
	"""
	Calculates the standard deviation on the dataset's given column

	Parameters:
	column (int array): the current column from the source dataset
	count (int): the number of elements in this column
	mean (float): the previously computed mean of the column's value

	Returns:
	(float): the standard deviation of the column
	"""
	std = 0
	for value in column:
		if not np.isnan(value):
			std += (value - mean) ** 2
		else:
			continue
	std /= count
	std = np.sqrt(std)
	return (std)

def fill_num_description(dataset):
	"""
	Creates and fills a DataFrame containing the necessary descriptions for each numerical column
	of the original dataset

	Parameters:
	dataset (pandas.DataFrame): the data structure obtained from the source file

	Returns:
	(pandas.DataFrame): the resulting description structure
	"""
	description = pd.DataFrame(index = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"], columns=dataset.columns)
	for column in dataset.columns:
		count = 0
		mean = 0
		tab = []
		for value in dataset[column]:
			if not np.isnan(value):
				count += 1
				mean += value
				tab.append(value)
			else:
				continue
		mean /= count
		description[column]["Count"] = count
		description[column]["Mean"] = mean
		description[column]["Std"] = compute_standard_deviation(dataset[column], count, mean)
		tab.sort()
		description[column]["Max"] = tab[count - 1]
		description[column]["Min"] = tab[0]
		description[column]["25%"] = tab[math.ceil(count / 4) - 1]
		description[column]["50%"] = tab[math.ceil((count + 1) / 2) - 1]
		description[column]["75%"] = tab[math.ceil(3 * count / 4) - 1]
	description.rename(columns=lambda c: c[:14], inplace=True)
	return (description)

def fill_alpha_description(dataset):
	"""
	Creates and fills a DataFrame containing the necessary descriptions for each non-numerical column
	of the original dataset

	Parameters:
	dataset (pandas.DataFrame): the data structure obtained from the source file

	Returns:
	(pandas.DataFrame): the resulting description structure
	"""
	description = pd.DataFrame(index = ["Count", "Unique", "Top", "Freq"], columns=dataset.columns)
	for column in dataset.columns:
		values = dataset[column].value_counts()
		description[column]["Count"] = len(dataset[column])
		description[column]["Unique"] = len(values)
		description[column]["Top"] = values.index[0]
		description[column]["Freq"] = values[0]
	return (description)

def display_dataset(dataset, type, more):
	"""
	Displays a description dataframe, adjusting the output if the "more" option was selected

	Parameters:
	dataset (pandas.DataFrame): the description of the data structure obtained from the source file
	type (string): the type of the description (numeric or alpha)
	more (boolean): whether the "more" option was selected or not

	Returns:
	(pandas.DataFrame): the resulting description structure
	"""
	print('\n')
	if (more):
		print("-" * 60)
		if (type == "numeric"):
			print("Numeric data:")
		else:
			print("Non-numeric data:")
		print("-" * 60)
	if (not dataset.empty):
		print(dataset)
	else:
		print("No " + type + " data!")

def main(argv):
	args = parse_arguments()
	dataset = read_source_data(args.data)
	num_dataset = trim_and_treat_data(dataset, 'numeric', args.normalize)
	num_description = fill_num_description(num_dataset)
	display_dataset(num_description, "numeric", args.more)
	if (args.more):
		alpha_dataset = trim_and_treat_data(dataset, 'alpha', False)
		alpha_description = fill_alpha_description(alpha_dataset)
		display_dataset(alpha_description, "alpha", args.more)

if __name__ == "__main__":
	main(sys.argv)