"""
Practice fitting an ML model to San Francisco 1990 Housing Data
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DATA_FILE = "datasets/housing/housing.csv"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")), DATA_FILE)
IMG_DIR = "images/end_to_end_project"
IMG_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")), IMG_DIR)


def main():
    pd.set_option("display.max_columns", None)
    housing = pd.read_csv(DATA_PATH, low_memory=False)
    # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing.income_cat):
        train_set = housing.loc[train_index]
        test_set = housing.loc[test_index]
    train_set.drop("income_cat", axis=1, inplace=True)
    test_set.drop("income_cat", axis=1, inplace=True)
    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()
    housing_prepared = prepare(housing)
    lm = fit_linear(housing_prepared, housing_labels)

def fit_linear(housing_prepared, housing_labels):
    model = LinearRegression()
    model.fit(housing_prepared, housing_labels)
    return model

def rmse(model, housing_prepared, housing_labels):
    predictions = model.predict(housing_prepared)
    return np.sqrt(mean_squared_error(housing_labels, predictions))

def plot_map(housing):
    housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,
        s=housing.population / 100,
        label="population",
        figsize=(10, 7),
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=True,
    )
    plt.legend()
    plt.savefig(os.path.join(IMG_PATH, "housing_latlong_scatter.png"))


def plot_hist(housing):
    attributes = [
        "median_house_value",
        "median_income",
        "total_rooms",
        "housing_median_age",
    ]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.savefig(os.path.join(IMG_PATH, "housing_scatter_matrix.png"))


def plot_onex_scatter(housing, x):
    housing.plot(kind="scatter", x=x, y="median_house_value", alpha=0.1)
    plt.savefig(os.path.join(IMG_PATH, f"housing_{x}_scatter.png"))


def impute_missing(housing):
    median_bedrooms = housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median_bedrooms, inplace=True)


def prepare(housing):
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    cat_attribs = ["ocean_proximity"]
    num_attribs = list(housing.drop(cat_attribs, axis=1))

    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),]
    )

    return full_pipeline.fit_transform(housing)

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_ix = 3
        bedrooms_ix = 4
        population_ix = 5
        households_ix = 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[
            X, rooms_per_household, population_per_household, bedrooms_per_room
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
