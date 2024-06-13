import os
import pickle
import mlflow
mlflow.set_tracking_uri("http://mlflow:5050")
mlflow.set_experiment("nyc-taxi-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):

    lr, dv = data

    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        mlflow.log_artifact(dv, "dict_vectorizer")
       
