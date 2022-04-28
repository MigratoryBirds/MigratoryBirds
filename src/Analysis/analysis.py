""" This file uses functions from autocorrelation.py to generate the real and random join counts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autocorrelation import join_counts,random_join_counts

df = pd.read_csv('resources/generated_data/joined_dataset.csv', index_col = False)

#seperating years
df19 = df[df.Year == 2019]
df20 = df[df.Year == 2020]
df21 = df[df.Year == 2021]

#setting index to be observation number
df19.index = range(df19.shape[0])
df20.index = range(df20.shape[0])
df21.index = range(df21.shape[0])

# IMPORTANT NOTE: Please make sure that the distance threshold for the real join counts and the random join counts is the same before comparing.

# Real Join Counts
JC19 = join_counts(df19,dis_threshold = 50)
JC20 = join_counts(df20,50)
JC21 = join_counts(df21,50)
print("2019 Join Counts: ",JC19,"\n2020 Join Counts: ",JC20,"\n2021 Join Counts: ",JC21)

# Randomized Join Counts
rand_JC19 = random_join_counts(df19,dis_threshold = 50,repetitions = 500)
rand_JC20 = random_join_counts(df20,50,500)
rand_JC21 = random_join_counts(df21,50,500)

def make_hist(list,bins,line_value,x_label):
    """ makes histogram with the x label being the number of join-counts in a simulation of certain type
        and the y label being the occurences across all simulations. plots the real value as a line
        so we can compare it against a random distribution"""
    plt.hist(list,bins = bins,ec = 'black')
    plt.ylabel('Counts')
    plt.xlabel(x_label)
    plt.axvline(x=line_value, color='r', linestyle='dashed', linewidth=2)

fig=plt.figure()

ax = fig.add_subplot(3,3,1)
make_hist(rand_JC19['AA'],bins = 10,line_value = JC19['AA'],x_label = 'AA 19')
ax = fig.add_subplot(3,3,2)
make_hist(rand_JC19['SS'],bins = 10,line_value = JC19['SS'],x_label = 'SS 19')
ax = fig.add_subplot(3,3,3)
make_hist(rand_JC19['AS'],bins = 10,line_value = JC19['AS'],x_label = 'AS 19')

ax = fig.add_subplot(3,3,4)
make_hist(rand_JC20['AA'],bins = 10,line_value = JC20['AA'],x_label = 'AA 20')
ax = fig.add_subplot(3,3,5)
make_hist(rand_JC20['SS'],bins = 10,line_value = JC20['SS'],x_label = 'SS 20')
ax = fig.add_subplot(3,3,6)
make_hist(rand_JC20['AS'],bins = 10,line_value = JC20['AS'],x_label = 'AS 20')

ax = fig.add_subplot(3,3,7)
make_hist(rand_JC21['AA'],bins = 10,line_value = JC21['AA'],x_label = 'AA 21')
ax = fig.add_subplot(3,3,8)
make_hist(rand_JC21['SS'],bins = 10,line_value = JC21['SS'],x_label = 'SS 21')
ax = fig.add_subplot(3,3,9)
make_hist(rand_JC21['AS'],bins = 10,line_value = JC21['AS'],x_label = 'AS 21')

fig.tight_layout()
plt.show()
