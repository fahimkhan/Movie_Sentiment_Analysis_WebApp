#!/usr/bin/env python

"""
Working with bigger data - Online alogorithms and out-of-core learning
"""

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

stop = stopwords.words('english')

def tokenizer(text):
	"""
	Remove the entire HTML markup
	"""
	text = re.sub('<[^>]*>','',text)
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
	text = re.sub('[\W]+', ' ', text.lower()) + \
	' '.join(emoticons).replace('-','')

	tokenized = [w for w in text.split() if w not in stop]
	
	return tokenized


def stream_docs(path):
	"""
	Reads in and return one document at a time
	"""
	with open(path,'r') as csv:
		next(csv) # Skip header
		for line in csv:
			text,label = line[:-3],int(line[-2])
			yield text,label

def get_minibatch(doc_stream,size):
	"""
	It take document stream from stream_docs function and return
	a particular number of documents specified by the size parameter
	"""
	docs,y = [],[]
	try:
		for _ in range(size):
			text,label = next(doc_stream)
			docs.append(text)
			y.append(label)
	except StopIteration:
		return None,None
	return docs,y


#Load data
df = pd.read_csv('./movie_data.csv')

#Vectorization for text processing. Use HashingVectorizer for memory usage optimization
vect = HashingVectorizer(decode_error='ignore',
	n_features=2**21,
	preprocessor=None,
	tokenizer=tokenizer)

clf = SGDClassifier(loss='log',random_state=1,n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')

#Lets perform out of core learning
import pyprind 
pbar = pyprind.ProgBar(45)
classes = np.array([0,1])
for _ in range(45):
	X_train,y_train = get_minibatch(doc_stream,size=1000)
	if not X_train:
		break
	X_train = vect.transform(X_train)
	clf.partial_fit(X_train,y_train,classes=classes)
	pbar.update()

#We have use the 1000*45 documents for training now lets use the last 5000 document for evaluating the performance
X_test,y_test = get_minibatch(doc_stream,size=5000)
X_test = vect.transform(X_test)
print ('Accuracy : %.3f' %clf.score(X_test,y_test))

#Now lets use the final 5000 documents to update our model
clf = clf.partial_fit(X_test,y_test)

##Serialized our classifier to store it in object file so that we can easily use it in our web application
import pickle
import os
#movieclassifier - where our web application file will be present
#pkl_objects - wherer our classifier and other objects will be stored
dest = os.path.join('movieclassifier','pkl_objects')
if not os.path.exists(dest):
	os.makedirs(dest)

pickle.dump(stop,open(os.path.join(dest,'stopwords.pkl'),'wb'),protocol=2)

pickle.dump(clf,open(os.path.join(dest,'classifier.pkl'),'wb'),protocol=2)

#For python3 you can use protocol=4 which is latest one


