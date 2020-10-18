# FIFA Player Market Value Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')

to_stay=[
    "ID","Name","Age","Nationality","Club","Overall","Potential",
    "Value","Wage","Real Face"
]

dataset.drop(dataset.columns.difference(to_stay),axis="columns",inplace=True)


dataset.set_index("ID",inplace=True)
dataset.isnull().sum()


# Convert Value column from string to number
dataset['Value2'] = dataset['Value'].apply(lambda x: x.split('€')[1])
dataset['Value3'] = dataset['Value2'].apply(
    lambda x: float(x.split('M')[0])*1000000 
    if x.split('M').__len__() > 1 else float(x.split('K')[0])*1000
)

dataset.drop(['Value2', 'Value'], axis="columns",inplace=True)
dataset.rename(columns={"Value3":"Value"}, inplace=True)


# Convert Wage column from string to number
dataset['Wage2'] = dataset['Wage'].apply(lambda x: x.split('€')[1])
dataset['Wage3'] = dataset['Wage2'].apply(
    lambda x: float(x.split('M')[0])*1000000 
    if x.split('M').__len__() > 1 else float(x.split('K')[0])*1000
)

dataset.drop(['Wage2', 'Wage'], axis="columns",inplace=True)
dataset.rename(columns={"Wage3":"Wage"}, inplace=True)
