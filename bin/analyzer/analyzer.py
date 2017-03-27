#!/usr/bin/env python3
# -*- coding: utf_8 -*-

import copy
from reader import train_set_reader, test_set_reader
from sklearn import linear_model
import numpy as np

class Analyzer:
    def __init__(self):
        ""
        self.train_set_reader = train_set_reader.TrainSetReader()
        self.test_set_reader = test_set_reader.TestSetReader()
        self.train_motocycles = self.train_set_reader.read()
        self.test_motocycles = self.test_set_reader.read()

    def analyze(self):
        reg = linear_model.LinearRegression()

        target = 'Bruto BPM'
        features = {
            'Catalogusprijs',
            'Massa ledig voertuig',
            'Wielbasis',
            'Aantal cilinders',
            'Cilinderinhoud'
        }

        train_means = self.train_motocycles.mean()
        self.train_motocycles = self.train_motocycles.fillna(train_means)
        self.test_motocycles = self.test_motocycles.fillna(train_means)

        # Train
        xvalues_train = list()
        yvalues_train = list()

        for index, motocycle in self.train_motocycles.iterrows():
            xvalues_train.append(list(motocycle[v] for v in features))
            yvalues_train.append(motocycle[target])

        regr = reg.fit(xvalues_train, yvalues_train)

        print('Coefficients: \n', regr.coef_)
        print('Intercept: \n', regr.intercept_)

        # Test
        xvalues_test = list()

        for index, motocycle in self.test_motocycles.iterrows():
            xvalues_test.append(list(motocycle[v] for v in features))

        results = regr.predict(xvalues_test)

        with open('output/results.csv', 'w') as output_file:
            output_file.write('%s,%s\n' % ('Kenteken', 'Prediction'))

            for index, motocycle in self.test_motocycles.iterrows():
                output_file.write('%s,%s\n' % (motocycle['Kenteken'], results[index]))
