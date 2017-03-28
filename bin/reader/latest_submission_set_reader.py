#!/usr/bin/env python3
# -*- coding: utf_8 -*-

import pandas as pd

class LatestSubmissionSetReader:
    def __init__(self):
        ""

    def read(self):
        return pd.read_csv('data/latest_submission.csv', sep=',')
