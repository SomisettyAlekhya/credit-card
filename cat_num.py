import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import logging
import sys

from log_file import phase_1
logger=phase_1("main")
from sklearn.preprocessing import OneHotEncoder
one_hot_ens=OneHotEncoder()
def conversion_cat_num(train_cat,test_cat):
    try:
        one_hot_ens.fit(train_cat[["Gender","Region"]])
        sol=one_hot_ens.transform(train_cat[['Gender','Region']]).toarray()
        train_cat[one_hot_ens.get_feature_names_out()[0]]=sol[:,0]
        train_cat[one_hot_ens.get_feature_names_out()[1]]=sol[:,1]
        train_cat[one_hot_ens.get_feature_names_out()[2]]=sol[:,2]
        train_cat[one_hot_ens.get_feature_names_out()[3]]=sol[:,3]
        train_cat[one_hot_ens.get_feature_names_out()[4]]=sol[:,4]
        train_cat[one_hot_ens.get_feature_names_out()[5]]=sol[:,5]
        train_cat[one_hot_ens.get_feature_names_out()[6]]=sol[:,6]

        one_hot_ens.fit(train_cat[["Gender", "Region"]])
        sol_1 = one_hot_ens.transform(test_cat[['Gender', 'Region']]).toarray()
        test_cat[one_hot_ens.get_feature_names_out()[0]] = sol_1[:, 0]
        test_cat[one_hot_ens.get_feature_names_out()[1]] = sol_1[:, 1]
        test_cat[one_hot_ens.get_feature_names_out()[2]] = sol_1[:, 2]
        test_cat[one_hot_ens.get_feature_names_out()[3]] = sol_1[:, 3]
        test_cat[one_hot_ens.get_feature_names_out()[4]] = sol_1[:, 4]
        test_cat[one_hot_ens.get_feature_names_out()[5]] = sol_1[:, 5]
        test_cat[one_hot_ens.get_feature_names_out()[6]] = sol_1[:, 6]

        train_cat=train_cat.drop(['Gender','Region'],axis=1)
        test_cat=test_cat.drop(['Gender','Region'],axis=1)
        logger.info(f"Categorical Features are converted into Numerical Columns")
        logger.info(f"By using One Hot Encoder")
        logger.info(train_cat.sample(5))
        #print(train_cat.sample(5))
        return train_cat,test_cat

    except Exception as e:
        er_type, er_msg, er_tb = sys.exc_info()
        line_number = er_tb.tb_lineno  # Extract the line number
        logger.error(f"Error from Line no: {line_number} Issue: {er_msg}")
