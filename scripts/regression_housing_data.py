from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tfl_training_anomaly_detection.config import get_config
from tfl_training_anomaly_detection.nb_utils import set_random_seed


class SKlearnModelProtocol(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


def get_numerical_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number, "float64", "int64"]).columns


def normalize_and_get_scaler(df: pd.DataFrame, columns: list["str"] = None):
    columns = columns or get_numerical_columns(df)
    scaler = StandardScaler()
    result = df.copy()
    result[columns] = scaler.fit_transform(df[columns])
    return result, scaler


def get_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(include=["object", "category"]).columns


def one_hot_encode_categorical(
    df: pd.DataFrame, columns: list[str] = None
) -> pd.DataFrame:
    columns = columns or get_categorical_columns(df)
    for column in columns:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df = df.drop(column, axis=1)
    return df


def train_sklearn_regression_model(
    model: SKlearnModelProtocol, df: pd.DataFrame, target_column: str
):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    model.fit(X, y)
    return model


def remove_nans(df: pd.DataFrame):
    # count rows with nans
    nans_count = df.isna().sum().sum()
    if nans_count > 0:
        print(f"Warning: {nans_count} NaNs were found and removed")
    return df.dropna()


def get_normalized_train_test_df(df: pd.DataFrame, test_size: float = 0.2):
    df = remove_nans(df)
    df, _ = normalize_and_get_scaler(df)
    df = one_hot_encode_categorical(df)
    train_df = df.sample(frac=1 - test_size)
    test_df = df.drop(train_df.index)
    return train_df, test_df


def evaluate_model(
    model: SKlearnModelProtocol, X_test: pd.DataFrame, y_test: pd.DataFrame
):
    y_pred = model.predict(X_test)
    print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
    print(f"R2 score: {r2_score(y_test, y_pred)}")
    return y_pred


if __name__ == "__main__":
    set_random_seed()
    c = get_config()
    housing_df = pd.read_csv(c.housing_data)

    # normalize and split data
    train_df, test_df = get_normalized_train_test_df(housing_df)

    # train
    trained_model = train_sklearn_regression_model(
        LinearRegression(), train_df, "median_house_value"
    )

    # evaluate
    y_pred = evaluate_model(
        trained_model,
        test_df.drop("median_house_value", axis=1),
        test_df["median_house_value"],
    )

    # visualize results
    plt.scatter(test_df["median_house_value"], y_pred)
    plt.xlabel("True median house value")
    plt.ylabel("Predicted median house value")
    plt.show()
