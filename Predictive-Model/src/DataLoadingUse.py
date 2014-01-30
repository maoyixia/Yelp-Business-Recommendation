import json
import numpy as np
import scipy as sp
from sklearn.feature_extraction import DictVectorizer
import pickle
import sys

f = open('business_consolidated_training.json')

file_list = f.readlines()

headers = ['review_count', 'stars', 'semantic_value', 'cluster_result']
numFeatures = 4
numInstances = len(file_list)

num_names = ['review_count', 'stars', 'semantic_value', 'cluster_result']
discard_names = []
cat_names=[]
target_names = ['open']

numericals=np.zeros((numInstances, len(num_names)))
discards = np.zeros((numInstances, len(discard_names)))
categoricals = []
target = np.zeros((numInstances, len(target_names)))

print 'Processing Data'
for i in range(0, numInstances):
	line = file_list[i]
	record = json.loads(line)
	business_id = record['business_id']

	# store target variables
	target[i][0] = record['open']

	
	# store numerical features
	numericals[i][0] = record['review_count']
	numericals[i][1] = record['stars']
	numericals[i][2] = record['semantic_value']
	numericals[i][3] = record['cluster_result']


#place numerical data into "dataMat" which may have categorical variables added to it
dataMat = numericals

#record header names
all_headers = num_names

#initialize a dict that will later store the number of values a categorical variable can take on
valDict = {}
for num_name in num_names:
    valDict[num_name] = 1 # numerical variables will be recorded as having one value
# note that this might cause confusion later
# if a categorical variable were to have
# only one value

## dataset is now in a list of dicts: can use DictVectorize to format categorical variables

if len(cat_names)>0:
    print 'Transforming Categorical Variables'

    vec = DictVectorizer()
    transformed = vec.fit_transform(categoricals).toarray()

    #print transformed[3]
    #sys.exit()

    #list of new categorical feature names
    featNames = vec.get_feature_names()

    print 'Forming final feature set'
    #concatenate numerical and categorical features
    dataMat = np.hstack((dataMat, transformed)) # the data matrix

    #concatenate numerical and categorical header names
    all_headers = all_headers+featNames

    #print "New feature set: ", all_headers
    #sys.exit()

    print 'Getting categorical feature values'
    #get the number of values for each categorical feature
    
    for i in xrange(len(featNames)):
        #get categorical variable names
        name = featNames[i].rstrip().split('=')[0]
        #this counts unique values for each variable name
        valDict[name] = valDict.setdefault(name,0)+1
        
#we can "serialize" files using pickle.  It's a lot faster than reloading from the csv and preprocessing every time.
def pickleIt(pyName, outputName):
    output = open(outputName+'.pk1', 'wb')
    pickle.dump(pyName, output)
    output.close()

pickleIt(dataMat, 'myDataMat')
pickleIt(target, 'myTarget') 
pickleIt(all_headers, 'myHeaders')
pickleIt(valDict, 'myValues')
pickleIt(num_names, 'numericalVariables')
pickleIt(cat_names, 'categoricalVariables')


print ' '
print 'Printing variable names and indices for future reference'
#print out variable names and indices for future reference
for i in xrange(len(all_headers)):
    print('Index number is '+repr(i)+' and feature name is '+repr(all_headers[i]))


