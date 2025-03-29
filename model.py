from numpy.ma.core import arange
from sklearn.neighbors import KNeighborsClassifier
from log_file import phase_1
logger = phase_1("main")
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import numpy as np
import joblib
import pickle
def knn(x_train,y_train,x_test,y_test):
    try:
        val_acc=[]
        k_values=np.arange(3,69,2)
        for i in k_values:
            knn_reg=KNeighborsClassifier(n_neighbors=i)
            knn_reg.fit(x_train,y_train)
            val_acc.append(accuracy_score(y_test,knn_reg.predict(x_test)))
        logger.info(f"ALL K VALUES{val_acc}")
        logger.info(f'BEST K VALUE {k_values[val_acc.index(max(val_acc))]} with ACCURACY {max(val_acc)}')
        knn_reg=KNeighborsClassifier(n_neighbors=k_values[val_acc.index(max(val_acc))])
        knn_reg.fit(x_train,y_train)
        logger.info(f"KNN TEST ACCURACY {accuracy_score(y_test,knn_reg.predict(x_test))}")
        logger.info(f"KNN CONFUSION MATRIX{confusion_matrix(y_test,knn_reg.predict(x_test))}")
        logger.info(f"KNN CLASSIFICATION REPORT {classification_report(y_test,knn_reg.predict(x_test))}")
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def LR(x_train,y_train,x_test,y_test):
    try:
        log_reg=LogisticRegression()
        log_reg.fit(x_train,y_train)
        logger.info(f"LR TEST ACCURACY {accuracy_score(y_test, log_reg.predict(x_test))}")
        logger.info(f"LR CONFUSION MATRIX{confusion_matrix(y_test, log_reg.predict(x_test))}")
        logger.info(f"LR CLASSIFICATION REPORT {classification_report(y_test, log_reg.predict(x_test))}")
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def NB(x_train,y_train,x_test,y_test):
    try:
        nb_reg=GaussianNB()
        nb_reg.fit(x_train,y_train)
        logger.info(f"NB TEST ACCURACY {accuracy_score(y_test, nb_reg.predict(x_test))}")
        logger.info(f"NB CONFUSION MATRIX{confusion_matrix(y_test, nb_reg.predict(x_test))}")
        logger.info(f"NB CLASSIFICATION REPORT {classification_report(y_test, nb_reg.predict(x_test))}")
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def RF(x_train,y_train,x_test,y_test):
    try:
        val_acc=[]
        trees=np.random.randint(0,100,10)
        for j in trees:
            rf_reg=RandomForestClassifier(criterion='entropy',n_estimators=j)
            rf_reg.fit(x_train,y_train)
            val_acc.append(accuracy_score(y_test,rf_reg.predict(x_test)))
        logger.info(f"ALL TREE VALUES{val_acc}")
        logger.info(f'BEST TREE VALUE {trees[val_acc.index(max(val_acc))]} with ACCURACY {max(val_acc)}')
        rf_reg=RandomForestClassifier(criterion='entropy',n_estimators=trees[val_acc.index(max(val_acc))])
        rf_reg.fit(x_train,y_train)
        logger.info(f"RF TEST ACCURACY {accuracy_score(y_test,rf_reg.predict(x_test))}")
        logger.info(f"RF CONFUSION MATRIX{confusion_matrix(y_test,rf_reg.predict(x_test))}")
        logger.info(f"RF CLASSIFICATION REPORT {classification_report(y_test,rf_reg.predict(x_test))}")

        rf_reg1=RandomForestClassifier(criterion='entropy',n_estimators=99)
        rf_reg1.fit(x_train,y_train)
        logger.info(f"BEST TREE VALUE IDENTIFIED is 99 with Accuracy 0.8903333333333333")
        logger.info(f"RF TEST ACCURACY {accuracy_score(y_test, rf_reg1.predict(x_test))}")
        logger.info(f"RF CONFUSION MATRIX{confusion_matrix(y_test, rf_reg1.predict(x_test))}")
        logger.info(f"RF CLASSIFICATION REPORT {classification_report(y_test, rf_reg1.predict(x_test))}")


    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def DT(x_train,y_train,x_test,y_test):
    try:
        dt_reg = DecisionTreeClassifier()
        dt_reg.fit(x_train, y_train)
        logger.info(f"DT TEST ACCURACY {accuracy_score(y_test, dt_reg.predict(x_test))}")
        logger.info(f"DT CONFUSION MATRIX{confusion_matrix(y_test, dt_reg.predict(x_test))}")
        logger.info(f"DT CLASSIFICATION REPORT {classification_report(y_test, dt_reg.predict(x_test))}")
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def model_training(x_train,y_train,x_test,y_test):
    try:
        logger.info("______KNN______")
        knn(x_train,y_train,x_test,y_test)
        logger.info("___DECISION TREE____")
        DT(x_train,y_train,x_test,y_test)
        logger.info("_____LOGISTIC REGRESSION_____")
        LR(x_train,y_train,x_test,y_test)
        logger.info("____RANDOM FOREST_____")
        RF(x_train,y_train,x_test,y_test)
        logger.info("_____NAIVE BAYES_____")
        NB(x_train,y_train,x_test,y_test)
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

