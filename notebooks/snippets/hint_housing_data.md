## Hint for Exercise 1

You should encode the categorical values and remove nans.

Then, create a train-test split, normalize the data frame and train
some sklear model, for example a linear regression, on the target column.

The results can be visualized with `plt.scatter`, you should also
compute some metrics like the mean squared error and the r2 score.

We found writing functions with the following signatures useful (you can
ignore the `Protocol` part, it's just a way to define a type hint):

```python

class SKlearnModelProtocol(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


def get_categorical_columns(df: pd.DataFrame) -> list["str"]:
    pass


def one_hot_encode_categorical(
    df: pd.DataFrame, columns: list[str] = None
) -> pd.DataFrame:
    pass


def train_sklearn_regression_model(
    model: SKlearnModelProtocol, df: pd.DataFrame, target_column: str
) -> SKlearnModelProtocol:
    pass


def remove_nans(df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_normalized_train_test_df(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    pass


def evaluate_model(
    model: SKlearnModelProtocol, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> np.ndarray:
    pass
```