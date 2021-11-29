import pandas as pd


class ContentBased:

    def __init__(self, x: pd.DataFrame):
        self._x = x
        self._user_profile: pd.DataFrame = pd.DataFrame()

    def fit(self) -> None:
        self._user_profile = self._x.sum(axis=0) / self._x.sum(axis=0).sum()

    def predict(self, x: pd.DataFrame) -> pd.Series:
        recommendation = ((self._user_profile * x).sum(axis=1))
        return recommendation.sort_values(ascending=False)
