from ny_utils import *

import datetime
import matplotlib.pyplot as plt
import os

# print difference between time in seconds
print t_dist('2010-06-17 19:04:10', '2010-06-17 20:28:16')

# read the data into np array
print read_NY_NY('Megan.csv')

# what day of the week was UC Berkeley founded?
print day_of_week('1868-03-23 14:00:00')

# loop over all training data files in the directory
for filename in os.listdir("./"):
  if filename.endswith(".csv"):
    if not filename.endswith("_quest.csv"):
      
      data = read_NY_NY(filename)
      
      #
      #  Run a very smart predictor
      #
      
      print 'Aha! I know what %s did last summer' % filename.split('.')[0]
      
       
        
# plot locations for a user, color in terms of the hour of the day and the day of the week
d = read_NY_NY('Megan.csv')

# get datetime objects from timestamps
datetime_labels = [datetime.datetime.fromtimestamp( lab ) for lab in d[:,2]]

# hour of the day
plt.subplot(1, 2, 1)
plt.scatter(d[:,1], d[:,0], 60, [h.hour for h in datetime_labels])
plt.title('Time of the day')
plt.colorbar()

# day of the week
plt.subplot(1, 2, 2)
plt.scatter(d[:,1], d[:,0], 60, [h.weekday() for h in datetime_labels])
plt.title('Day of the week')
plt.colorbar()

plt.show()