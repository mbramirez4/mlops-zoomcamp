import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def train(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("taxi-trip-duration")

    categorical = ["PULocationID", "DOLocationID"]

    df[categorical] = df[categorical].astype("str")
    train_dicts = df[categorical].to_dict(orient="records")

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df["duration"].values

    model = LinearRegression()
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        run_id = run.info.run_id

    return {"dv": dv, "model": model, "run_id": run_id}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'