import numpy as np
import pickle 

from matplotlib import pyplot as plt
from pylab import *


### this will be replaced with the real test image ###
im_test = plt.imread('parking_test_preview.png')
###


# This function MUST take locations (loc) and an image (im) 
# as input parameters and return the feature vector
def my_feature_vector(loc, im, size = 10):
  w = size
  # a patch of the size w cenetered at loc is extracted as a feature vector
  patch = im[loc[1]-w:loc[1]+w, loc[0]-w:loc[0]+w]
  p = np.array(patch).flatten()
  return p 
  

## 10 preview test locations
### these will be replaced with the real set of 100 test locations ###
test_locs_labs = np.load('test_locations_and_labels_preview.np')

test_locs   = test_locs_labs[:,0:2]
test_labels = test_locs_labs[:,2]

X_test = []
for loc in test_locs:
  X_test.append( my_feature_vector(loc, im_test) )

my_classifier = pickle.load(open('classifier_yiyange.pickle')) 

score = 0
for i, xtest in enumerate(X_test): 
  
  predicted = my_classifier.predict(xtest)
  
  if (test_labels[i] == 1.0)&(predicted == 1.0):
     score = score + 2
  
  if (test_labels[i] == 1.0)&(predicted == 0.0):
     score = score - 0.5
  
  if (test_labels[i] == 0.0)&(predicted == 1.0):
     score = score - 0.5
  
  if (test_labels[i] == 0.0)&(predicted == 0.0):
     score = score + 0.25
     
  print (test_labels[i], predicted, score)

print ('You final Score is: %.2f' % score)
