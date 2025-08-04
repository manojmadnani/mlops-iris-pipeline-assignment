# src/utils.py
from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    return df
