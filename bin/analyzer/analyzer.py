#!/usr/bin/env python3
# -*- coding: utf_8 -*-

import copy
from reader import train_set_reader, test_set_reader
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
import csv
from math import sqrt


class Analyzer:
    def __init__(self):
        ""
        self.train_set_reader = train_set_reader.TrainSetReader()
        self.test_set_reader = test_set_reader.TestSetReader()
        self.train_motocycles = self.train_set_reader.read()
        self.test_motocycles = self.test_set_reader.read()
        self.today = pd.to_datetime('today')

    def calculate_days(self, motocycle):
        date_admission = pd.to_datetime(motocycle['Datum eerste toelating'])
        # date_ascription = pd.to_datetime(motocycle['Datum tenaamstelling'])

        motocycle['Dagen sinds eerste toelating'] = self.today - date_admission
        # motocycle['Dagen sinds tenaamstelling'] = self.today - date_ascription

        return motocycle

    def analyze(self, calculate_rmse=False):
        # base = ensemble.GradientBoostingRegressor()
        base = ensemble.RandomForestRegressor()
        # base = linear_model.LinearRegression()

        reg = ensemble.AdaBoostRegressor(base, loss='square')

        # Train
        self.train_motocycles.dropna(inplace=True)
        df_train = self.train_motocycles[[
            'Merk',
            'Catalogusprijs',
            'Massa ledig voertuig',
            'Wielbasis',
            'Aantal cilinders',
            'Cilinderinhoud',
            'Bruto BPM'
        ]].reindex()

        print(df_train.info())

        df_test = self.test_motocycles[[
            'Kenteken',
            'Merk',
            'Catalogusprijs',
            'Massa ledig voertuig',
            'Wielbasis',
            'Aantal cilinders',
            'Cilinderinhoud'
        ]].reindex()
        df_test.fillna(df_test.mean(), inplace=True)

        # Prepare
        vec = DictVectorizer()
        vec.fit(df_train.to_dict(orient='records'))
        vec.fit(df_test.to_dict(orient='records'))

        y_train = df_train.pop('Bruto BPM')
        X_train = vec.transform(df_train.to_dict(orient='records'))

        number_plates = df_test.pop('Kenteken')
        X_test = vec.transform(df_test.to_dict(orient='records'))

        # X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Fit model
        reg.fit(X_train, y_train)

        y_predict = reg.predict(X_test)

        # if calculate_rmse:
        #     print('Root Mean Squared Error: \n', sqrt(mean_squared_error(y_test, y_predict)))

        with open('output/results.csv', 'w') as output_file:
            rows = zip(number_plates, y_predict)

            w = csv.writer(output_file)
            w.writerow(['Kenteken', 'Prediction'])

            for row in rows:
                w.writerow([row[0], row[1]])
