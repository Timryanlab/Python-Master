# -*- coding: utf-8 -*-
"""
Title:

Concept:
Created on Sat Aug 22 09:48:29 2020

@author: Andrew Nelson
"""
#% Import Libraries
import pandas as pd

#% Functions

#% Main Workspace
if __name__ == '__main__':
    expenses = pd.read_csv('Accounting_Westfall_expenses.csv')
    canceled = pd.read_csv('Accounting_Westfall_canceled.csv')
    expired = pd.read_csv('Accounting_Westfall_expired.csv')
    sales = pd.read_csv('Accounting_Westfall_sales.csv')
