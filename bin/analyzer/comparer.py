#!/usr/bin/env python3
# -*- coding: utf_8 -*-

from math import sqrt
from reader import latest_submission_set_reader, output_set_reader
from sklearn.metrics import mean_squared_error
import pandas as pd

class Comparer:
    def __init__(self):
        ""
        self.latest_submission_set_reader = latest_submission_set_reader.LatestSubmissionSetReader()
        self.latest_submission = self.latest_submission_set_reader.read()
        self.output_set_reader = output_set_reader.OutputSetReader()
        self.output = self.output_set_reader.read()
        self.today = pd.to_datetime('today')

    def compare(self):
        latest_predictions = list()

        for index, motocycle in self.latest_submission.iterrows():
            latest_predictions.append(motocycle['Prediction'])

        predictions = list()

        for index, motocycle in self.output.iterrows():
            predictions.append(motocycle['Prediction'])

        rmse = sqrt(mean_squared_error(latest_predictions, predictions))
        print('Mean Squared Error: \n', rmse)
