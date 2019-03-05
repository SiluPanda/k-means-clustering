from copy import deepcopy
from itertools import cycle
from pprint import pprint as pprint
import sys
import argparse
import matplotlib.pyplot as plt
import random
import math


def distance_euclidean(p1, p2):
	'''
	p1: tuple: 1st point
	p2: tuple: 2nd point

	Returns the Euclidean distance b/w the two points.
	'''

	distance = None

	# TODO [task1]:
	# Your function must work for all sized tuples.

	########################################
	sqrd_sum = 0.0
	for i in range(len(p1)):
		sqrd_sum += (p1[i] - p2[i])**2

	distance = math.sqrt(sqrd_sum)
	return distance

def kmeans_iteration_one(data, centroids):
	'''
	data: list of tuples: the list of data points
	centroids: list of tuples: the current cluster centroids
	distance: callable: function implementing the distance metric to use

	Returns a list of tuples, representing the new cluster centroids after one iteration of k-means clustering algorithm.
	'''

	new_centroids = []


	# TODO [task1]:
	# You must find the new cluster centroids.
	# Perform just 1 iteration (assignment+updation) of k-means algorithm.

	########################################
	categorizing_points = []
	for i in range(len(centroids)):
		categorizing_points.append([])

	for i in range(len(data)):
		all_distances_to_centroid = []
		for j in range(len(centroids)):
			all_distances_to_centroid.append(distance_euclidean(centroids[j], data[i]))

		index_of_nearest = all_distances_to_centroid.index(min(all_distances_to_centroid))
		categorizing_points[index_of_nearest].append(data[i])

	def mean_of_tuple_list(list_of_tuple):
		ans = []
		for i in range(len(list_of_tuple[0])):
			avg = 0.0
			for j in range(len(list_of_tuple)):
				avg = avg + list_of_tuple[j][i]
			avg = avg / len(list_of_tuple)

			ans.append(avg)

		return tuple(ans)

	for i in range(len(categorizing_points)):
		new_centroids.append(mean_of_tuple_list(categorizing_points[i]))





	assert len(new_centroids) == len(centroids)
	return new_centroids

print(kmeans_iteration_one([(1, 0.0), (2, 2.3), (3, 4.6), (4, 6.8999999999999995), (5, 9.2)],[(5, 1), (-1, 2), (3, 6)] ))
