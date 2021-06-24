# 42 Machine learning project: DSLR

DSLR is the intermediate project in 42's machine learning branch. It introduces two new concepts: **logistic regression**, and **data visualisation**.

We are given a dataset, consisting of 1600 Hogwarts students's various characteristics (i.e their grades in all classes, birth date, names, etc..) and their *House*.
It's a classification problem, the goal is to train a model that will be able to predict a student's House, among the 4 Hogwarts Houses, from his characteristics.

The assignment has multiple parts: first we must visualise our data, to help us clean it and select the useful characteristics for our training. Then we can train our
model on the dataset in **dataset_train.csv**, and finally, we can use it to predict the Houses of the students described in **dataset_test.csv**.

## Specifications

* The whole project was coded using python 3.6.9, and uses the matplotlib, pandas and numpy libraries
* Datasets are stored in csv files
* There are six different programs: describe.py, histogram.py, scatter_plot.py, pair_plot.py, logreg_train.py and logreg_predict.py
* Obviously logreg_train.py should be run before logreg_predict.py

## describe

This simply describes the given dataset's various features, with an output that is similar to ``pandas.describe()``.
It can be run with ``-m`` (or ``--more``) to describe non-numerical data as well; and ``-n`` to normalize the numerical features (i.e the student's grades)

## histogram

Plots histograms for every class, displaying the students' grade repartition in different colors, according to their House. 

## scatter_plot

Displays scatter plots between every possible pairing of two classes, to show the correlation between them (There are 13 classes, so that means 169 different plots).

## pair_plot

Displays the grades' pair plot, which is basically a combination of histograms and scatter_plots.

## logreg_train

From the previous **data visualisation**, we selected five different classes to train our model as best we can: **Herbology, Divination,
Ancient Runes, Charms** and **Defense Against the Dark Arts.** The model is trained using logistic regression, with a one-vs-all approach since there are four different Houses.
The resulting weights are stored, by default, in **weights.npy**

The algorithm's learning rate can be specified with ``-l`` and the number of iterations with ``-i``. A verbose option, ``-v`` allows us to see the elapsed time and the
final score of our model on the training dataset.

## logreg_predict

From the previously computed weights stored in ``weights.npy``, this program will predict the Houses of the House-less students described in **dataset_test.csv**. The resulting
Houses, associated with the students' indexes, are stored in ``houses.csv``

Final score compared to the "true" Houses of the students in **dataset_truth.csv** is 99%.

## Example of use

```
$> python logreg_train.py -v 
Training finished in 0.452s. Score obtained: 0.983
Weights saved to weights.npy
$> python logreg_predict.py
Resulting houses saved in houses.csv
$> cat houses.csv
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
2,Gryffindor
3,Hufflepuff
4,Hufflepuff
5,Slytherin
...
399,Ravenclaw
```
