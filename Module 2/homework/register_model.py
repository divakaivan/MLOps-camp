import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        param_types = {
        'bootstrap': bool,
        'ccp_alpha': float,
        'criterion': str,
        'max_depth': int,
        'max_features': float,
        'max_leaf_nodes': int,
        'max_samples': float,
        'min_impurity_decrease': float,
        'min_samples_leaf': int,
        'min_samples_split': int,
        'min_weight_fraction_leaf': float,
        'monotonic_cst': None,
        'n_estimators': int,
        'n_jobs': int,
        'oob_score': bool,
        'random_state': int,
        'verbose': int,
        'warm_start': bool
    }
        params = run.data.params
    # Convert params to their appropriate types
        for param, param_type in param_types.items():
            if param in params:
                value = params[param]
                if param_type == bool:
                    params[param] = value == 'True'
                elif param_type == int:
                    if value.lower() == 'none':
                        params[param] = None
                    else:
                        params[param] = int(value)
                elif param_type == float:
                    if value.lower() == 'none':
                        params[param] = None
                    else:
                        params[param] = float(value)
                elif param_type == str:
                    params[param] = value
                elif param_type is None:
                    params[param] = None  # For parameters that can be None
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type} for parameter: {param}")

        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )[-1]

    # Register the best model
    mlflow.register_model(f"runs:/{best_run.info.run_id}/model", "best_rf_model")


if __name__ == '__main__':
    run_register_model()