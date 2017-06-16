import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

def t_dist(str1, str2):
   
  t1 = datetime.datetime.strptime(str1, '%Y-%m-%d %H:%M:%S')
  t2 = datetime.datetime.strptime(str2, '%Y-%m-%d %H:%M:%S')
   
  if t2>t1:
    return (t2-t1).total_seconds()
  else:
    return (t1-t2).total_seconds()
     
     
def read_NY_NY(user_name):

  d, timestr = [], []
  with open(user_name,'r') as f:
    for line in f:  
      l = line.strip().split(',')
    
      timestr.append( datetime.datetime.strptime(l[0], '%Y-%m-%d %H:%M:%S') )
      d.append([float(l[1]), float(l[2]), t_dist(l[0], '1970-01-01 00:00:00') ]) 
    
  return np.array(d)
  

def day_of_week(str1):

  return datetime.datetime.strptime(str1, '%Y-%m-%d %H:%M:%S').weekday()