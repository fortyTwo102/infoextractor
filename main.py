import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from itertools import product
import matplotlib.patches as mpatches
import sys

def is_categorical(feature):
    return feature.nunique() < 50


def plot_against(data, categorical, continuous, target):
	# this plots categorical features against a continuos feature

	print("\n\n\n------- plotting", categorical, '_vs_', continuous, '-------\n\n\n\n')
	sns.catplot(x=categorical, y=continuous, data=data, hue=target, kind = 'boxen')
	
	foldername = 'categorical_vs_continuous'
	
	path = os.path.join(os.getcwd(), sys.argv[1][:-4])

	try:
		os.makedirs(path)
	except:
		pass

	path = os.path.join(path, foldername)

	try:
		os.makedirs(path)
	except:
		pass
    
	filename =  categorical + '_vs_' + continuous + '.jpg'
	filepath = os.path.join(path, filename)
	plt.savefig(filepath)


def plot_categorical(data, feature, target):
	# this plots categorical features individually

	print("\n\n\n------- plotting", feature, '-------\n\n\n\n')
	sns.countplot(x=feature, data=data, hue = target)
	
	foldername = 'categorical'
	path = os.path.join(os.getcwd(), sys.argv[1][:-4])

	try:
		os.makedirs(path)
	except:
		pass
		
	path = os.path.join(path, foldername)

	try:
		os.makedirs(path)
	except:
		pass
    
	filename = feature + '.jpg'
	filepath = os.path.join(path, filename)
	plt.savefig(filepath)

def plot_continuous2(data, feature, target):
	# this plots categorical features individually


	from scipy import stats

	print(data.shape)
	z = np.abs(stats.zscore(data))
	data = data[(z < 2).all(axis=1)]
	print(data.shape)


	print("\n\n\n------- plotting", feature, '-------\n\n\n\n')
	sns.distplot(data[feature], kde = False, bins = 4)
	
	foldername = 'continuous'
	path = os.path.join(os.getcwd(), sys.argv[1][:-4])

	try:
		os.makedirs(path)
	except:
		pass
		
	path = os.path.join(path, foldername)

	try:
		os.makedirs(path)
	except:
		pass
    
	filename =  feature + '.jpg'
	filepath = os.path.join(path, filename)
	plt.savefig(filepath)

def plot_continuous(data, feature, target):
	# this plots continuous features individually

	print("\n\n\n------- plotting", feature, '-------\n\n\n\n')
	unique_targets = data[target].unique()
	divided_dataset = []
	feature_list = []
	rand_colors = []

	target_iter = iter(unique_targets)

	for target_value in unique_targets:
		temp = data[data[target] == target_value]
		divided_dataset.append(temp)

	for dataset in divided_dataset:
		feature_list.append(dataset[feature])
	
	fig, ax = plt.subplots()

	patches = []

	for dataset in divided_dataset:
		rand_color = np.random.rand(3,)
		rand_colors.append(rand_color)
		patch = mpatches.Patch(color=rand_color, label=dataset.iloc[-1,-1])
		patches.append(patch)

	rand_color_iter = iter(rand_colors)

	plt.legend(handles=patches)

	# bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

	for feature_name in feature_list:
		color = np.random.rand(3,)
		sns.distplot(feature_name, ax = ax, kde = False, hist_kws = {'color':next(rand_color_iter)}, label = "hey")

	foldername = 'continuous'
	path = os.path.join(os.getcwd(), sys.argv[1][:-4])

	try:
		os.makedirs(path)
	except:
		pass
		
	path = os.path.join(path, foldername)

	try:
		os.makedirs(path)
	except:
		pass
    
	filename =  feature + '.jpg'
	filepath = os.path.join(path, filename)
	plt.savefig(filepath)	



filepath = sys.argv[1]
target = sys.argv[2]


data = pd.read_csv(filepath)

# unique_targets = data[target].unique()
# # divided_dataset = []

# from scipy import stats

# print(data.shape)
# z = np.abs(stats.zscore(data))
# data = data[(z < 2).all(axis=1)]
# print(data.shape)


cate = []
cont = []

# for value in unique_targets:
# 	divided_dataset.append(data[data[target] == value])


# plot_1(data, 'key', 'loudness', target)
# print(data.head())


for feature in data.columns:

	if feature == target:
		pass

	else:	

		if is_categorical(data[feature]):
			print("cat: ", feature)
			cate.append(feature)

		else:
			print("cont: ", feature)
			cont.append(feature)


# for combo in product(cate, cont):
# 	plot_against(data, combo[0], combo[1], target)

for feature in cate:
	plot_categorical(data, feature, target)

# for feature in cont:
# 	plot_continuous2(data, feature, target)

