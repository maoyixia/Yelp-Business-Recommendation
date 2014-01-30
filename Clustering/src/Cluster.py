## Author: Yixia Mao

import math
import pickle
import operator
import numpy as np
import json

# size of cluster
k = 100

def pickleLoad(inputName):
    pk1_file = open(inputName+'.pk1', 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj

name = pickleLoad('myName')
location = pickleLoad('myLocation')
category = pickleLoad('myCategory')
star = pickleLoad('myStar')

# Calculate distance of two locations according to long/lat
def distanceHelper(location1, location2):
	dis = math.pow((location1[0] - location2[0]), 2) + math.pow((location1[1] - location2[1]), 2)
	return dis

# Store every distance pairs of all the locations
distance = {}
for i in location.keys():
	dis = {}
	for j in location.keys():
		dis[j] = distanceHelper(location[i], location[j])
	distance[i] = dis

# Find top k nearest neighbors and form clusters of same categories
for i in distance.keys():
	sorted_dis = dict(sorted(distance[i].iteritems(), key=lambda x: x[1])[:(k+1)])
	neighbors = []
	targetCategory = category[i]
	for j in sorted_dis.keys():
		if j != i:
			for c in category[j]:
				if c in targetCategory:
					neighbors.append(j)	
					break
	distance[i] = neighbors

# Grade every business according to its neighbors' stars. Can change grading scheme here
grade = {}
for i in distance.keys():
	stars = []
	for j in distance[i]:
		stars.append(star[j])
	if len(stars) == 0:
		score = 0
	else:
		mean = np.mean(stars, axis=0)
		score = (star[i] - mean) / mean
	grade[i] = score

# convert dict to JSON and write JSON to file
with open('result.json', 'w') as outfile:
	grade_json = {}
	for i in grade.keys():
		grade_json["business_id"] = i
		grade_json["cluster_result"] = grade[i]
  		json.dump(grade_json, outfile)
  		outfile.write('\n')



















