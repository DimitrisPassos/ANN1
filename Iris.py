from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer as cancer
from sklearn.datasets import load_iris as iris
import pandas as pd


iris = iris(as_frame=False, return_X_y=True)