import os
import pickle

import mlflow

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")

    model = data["model"]
    dv = data["dv"]
    run_id = data["run_id"]
    
    with mlflow.start_run(run_id=run_id):
        mlflow.sklearn.log_model(model, "model")

        with open("dv.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("dv.pkl", "dv")

    os.remove("dv.pkl")


