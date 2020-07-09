# -*- coding: utf-8 -*-
"""
Title: Visualizing Covid Data

Concept: A script that allows me to grab Covid19 data from Johns Hopkins so I 
can start looking at trends myself
Created on Sun Jul  5 09:57:26 2020

@author: Andrew Nelson
"""
#% Import Libraries
import requests
from datetime import datetime as dt
import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
#% Functions
def load_covid_data(): # Load most up to date covid data to pandas data frame
    file_name = 'covid_19.csv'
    global_csv_url = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'
    us_county_csv_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    us_county_csv_url = 'https://covidtracking.com/api/v1/states/daily.csv'
    get_covid_data(global_csv_url)
    global_data_frame = load_from_csv_to_pandas(file_name)
    get_covid_data(us_county_csv_url)
    us_county_data_frame = load_from_csv_to_pandas(file_name)
    european_union = pd.Index(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark',
                          'Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy','Latvia',
                          'Lithuania','Luxembourg','Matla','Netherlands','Poland','Portugal',
                          'Romania','Slovakia','Slovenia','Spain','Sweden'])
    return global_data_frame, us_county_data_frame # Return pandas data frame

def load_from_csv_to_pandas(file_name):
    covid_data_frame = pd.read_csv(file_name)
    return covid_data_frame
    
def get_covid_data(csv_url):
    response = requests.get(csv_url)
    file_name = 'covid_19.csv'
    csv_file = open(file_name, 'wb')
    csv_file.write(response.content)
    csv_file.close()



def previous_day_average(x, day = 5):
    y = np.empty_like(x)
    for i in range(y.shape[0]):
        if i < day:
            y[i] = x[:i].mean()
        else:
            y[i] = x[i-day:i].mean()
    return y




#% Main Workspace
if __name__ == '__main__':
    global_data_frame, us_county_data_frame = load_covid_data()
    
    
    
    # Show data example
    us_county_data_frame.head()
    
    # Reorganize by date and get 'string date'
    us_county_data_frame = us_county_data_frame.iloc[::-1]
    us_county_data_frame[['str_date']] = us_county_data_frame[['date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[4:6],s[6:],s[:4]))
    us_county_data_frame.set_index('date')
    # Let's try to grab one state and plot it's covid history
    states = us_county_data_frame.state.unique() # Grab a list of all states in the data set
    dates = us_county_data_frame.str_date.unique()
    
    
    state = 'NY'
    state_frames = us_county_data_frame.loc[us_county_data_frame['state'] == state]
    # State frames is now a data frame comprised of 1 state's data
    state_frames['daily_positives'] = np.diff(state_frames['positive'], prepend = 0)
    state_frames['daily_negatives'] = np.diff(state_frames['negative'], prepend = 0)
    state_frames['daily_tests'] = np.diff(state_frames['total'], prepend = 0)
    state_frames['daily_positive_per_test'] = state_frames['daily_positives']/state_frames['daily_tests']
    ax = state_frames.plot(x = 'str_date', 
                           y = 'daily_positive_per_test', 
                           label = 'Daily % of positive tests')

    ax2 = state_frames.plot(ax = ax,  
                           x = 'str_date', 
                           y = 'daily_tests', 
                           label = 'Number of Tests performed',
                           secondary_y = True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax.right_ax.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='top')
    pl.suptitle('Daily Positive Cases versus Total Test Performed in ' + state +' (Daily)')
    dates = state_frames['str_date']

    ax.set_xticks([0, int(dates.shape[0]/4), int(dates.shape[0]/2),int(3*dates.shape[0]/4), dates.shape[0]])
    ax.set_xticklabels([dates.iloc[0], dates.iloc[int(dates.shape[0]/4)], dates.iloc[int(dates.shape[0]/2)],dates.iloc[int(3*dates.shape[0]/4)], dates.iloc[-1]])
    ax.set_xlabel('Date')
    ax.set_ylim(0,1)
    ax.set_ylabel('% Positive Tests')
    ax2.set_ylabel('Total Tests Performed')