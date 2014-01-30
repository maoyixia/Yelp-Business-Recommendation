## Author: Yixia Mao

import json
import pickle

f = open('yelp_training_set_business.json')

file_list = f.readlines()

numInstances = len(file_list)

name = {}
location = {}
category = {}
star = {}

for i in range(0, numInstances):
	line = file_list[i]
	record = json.loads(line)
	business_id = record['business_id']

	# store map: <business_id, business_name>
	name[business_id] = record['name']

	# store map: <business_id, [longitude, latitude]>
	geo = [record['longitude'], record['latitude']]
	location[business_id] = geo

	# store map: <business_id, categories>
	category[business_id] = record['categories']

	# store map: <business_id, star>
	star[business_id] = record['stars']

#we can "serialize" files using pickle.  It's a lot faster than reloading from the csv and preprocessing every time.
def pickleIt(pyName, outputName):
    output = open(outputName+'.pk1', 'wb')
    pickle.dump(pyName, output)
    output.close()

print 'Serializing'
pickleIt(name, 'myName')
pickleIt(location, 'myLocation')
pickleIt(category, 'myCategory')
pickleIt(star, 'myStar')



