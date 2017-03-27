#!/usr/bin/env python3
# -*- coding: utf_8 -*-

import copy
from reader import train_set_reader, test_set_reader
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import csv
from math import sqrt

class Analyzer:
    def __init__(self):
        ""
        self.train_set_reader = train_set_reader.TrainSetReader()
        self.train_motocycles = self.train_set_reader.read()
        self.today = pd.to_datetime('today')

    def analyze_train_set(self):
        self.analyze(self.train_motocycles)

    def analyze_test_set(self):
        test_motocycles = test_set_reader.TestSetReader().read()
        self.analyze(test_motocycles)

    def calculate_days(self, motocycle):
        date_admission = pd.to_datetime(motocycle['Datum eerste toelating'])
        # date_ascription = pd.to_datetime(motocycle['Datum tenaamstelling'])

        motocycle['Dagen sinds eerste toelating'] = self.today - date_admission
        # motocycle['Dagen sinds tenaamstelling'] = self.today - date_ascription

        return motocycle

    def analyze(self, df):
        reg = ensemble.GradientBoostingRegressor(max_depth=16)

        target = 'Bruto BPM'
        features = {
            'Catalogusprijs',
            'Massa ledig voertuig',
            'Wielbasis',
            'Aantal cilinders',
            'Cilinderinhoud',
            'Dagen sinds eerste toelating'
        }

        # Fill empty values
        train_means = self.train_motocycles.mean()
        self.train_motocycles.fillna(train_means, inplace=True)
        df.fillna(train_means, inplace=True)

        min_date = pd.to_datetime(df['Datum eerste toelating']).min()
        df['Datum eerste toelating'].fillna(min_date, inplace=True)

        print(df.info())

        # Train
        xvalues_train = list()
        yvalues_train = list()

        for index, motocycle in self.train_motocycles.iterrows():
            motocycle = self.calculate_days(motocycle)

            xvalues_train.append(list(motocycle[v] for v in features))
            yvalues_train.append(motocycle[target])

        regr = reg.fit(xvalues_train, yvalues_train)

        # Test
        xvalues_test = list()

        for index, motocycle in df.iterrows():
            motocycle = self.calculate_days(motocycle)

            xvalues_test.append(list(motocycle[v] for v in features))

        results = regr.predict(xvalues_test)

        print('Mean Squared Error: \n', sqrt(mean_squared_error(yvalues_train, results)))

        with open('output/results.csv', 'w') as output_file:
            w = csv.writer(output_file)
            w.writerow(['Kenteken', 'Prediction'])

            for index, motocycle in df.iterrows():
                w.writerow([motocycle['Kenteken'], results[index]])
