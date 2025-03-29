import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import logging
import sys

from log_file import phase_1
logger=phase_1("main")
from sklearn.preprocessing import OrdinalEncoder
one_hot_eno=OrdinalEncoder()
def conversion_cat_num1(train_set,test_set):
    try:
        one_hot_eno.fit(train_set[['Rented_OwnHouse','Occupation','Education']])
        sol=one_hot_eno.transform(train_set[['Rented_OwnHouse','Occupation','Education']])
        train_set[one_hot_eno.get_feature_names_out()[0]]=sol[:,0]
        train_set[one_hot_eno.get_feature_names_out()[1]]=sol[:,1]
        train_set[one_hot_eno.get_feature_names_out()[2]]=sol[:,2]

        sol_1 = one_hot_eno.transform(test_set[['Rented_OwnHouse', 'Occupation', 'Education']])
        test_set[one_hot_eno.get_feature_names_out()[0]] =sol_1[:, 0]
        test_set[one_hot_eno.get_feature_names_out()[1]] = sol_1[:, 1]
        test_set[one_hot_eno.get_feature_names_out()[2]] = sol_1[:, 2]

        logger.info(f"Ordinal Features are converted to numerical columns successfully")
        return train_set,test_set

    except Exception as e:
        er_type, er_msg, er_tb = sys.exc_info()
        line_number = er_tb.tb_lineno  # Extract the line number
        logger.error(f"Error from Line no: {line_number} Issue: {er_msg}")
