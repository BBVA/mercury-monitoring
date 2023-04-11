import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from mercury.monitoring.estimation.performance_predictor import PerformancePredictor
from mercury.dataschema import DataSchema 

np.random.seed(seed=2021)


def test_performance_predictor_binary_classification():
    
    # Create Binary Classification Dataset
    
    X, y = make_classification(n_samples=1000, n_features=3, n_informative=3, n_redundant=0, random_state=15)
    X = pd.DataFrame(X, columns=["f" + str(i) for i in range(X.shape[1])])
    X_source, X_serving, y_source, y_serving = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Split Serving and add Drift
    X_serving_1 = X_serving.iloc[0:250].copy()
    X_serving_2 = X_serving.iloc[250:].copy()
    y_serving_1 = y_serving[0:250]
    y_serving_2 = y_serving[250:]


    feature = "f0"
    p = 0.5
    amount = 10.5
    noise = 0.05
    feature_values = X_serving_2[feature].values
    idx_corrupt = np.random.choice(range(len(feature_values)), size=int(len(feature_values) * p), replace=False)
    shift_values = np.random.normal(loc=amount, scale=noise, size=len(idx_corrupt))
    feature_values[idx_corrupt] += shift_values
    X_serving_2[feature] = feature_values

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X_source, y_source, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model = model.fit(X_train, y_train)
    
    # Define Data Corruptions for Performance predictor
    corruptions = [
        ('shift_drift', {'cols': ['f0'], 'force': 0.0, 'noise': 0.01}),
        ('shift_drift', {'cols': ['f0'], 'force': 0.2, 'noise': 0.05}),
        ('shift_drift', {'cols': ['f0'], 'force': 0.5, 'noise': 0.05}),
        ('shift_drift', {'cols': ['f0'], 'force': 1.0, 'noise': 0.1}),
        ('shift_drift', {'cols': ['f0'], 'force': 1.5, 'noise': 0.1}),
        ('shift_drift', {'cols': ['f0'], 'force': 2.0, 'noise': 0.1}),
        ('shift_drift', {'cols': ['f0'], 'force': 5.0, 'noise': 0.1}),
        ('shift_drift', {'cols': ['f0'], 'force': 10.0, 'noise': 0.1}),
        ('shift_drift', {'cols': ['f0'], 'force': 20.0, 'noise': 0.1}),
        ('scale_drift', {'cols': ['f0'], 'mean': 0.1, 'sd': 0.01}),
        ('scale_drift', {'cols': ['f0'], 'mean': 0.5, 'sd': 0.01}),
        ('scale_drift', {'cols': ['f0'], 'mean': 0.7, 'sd': 0.01}),
        ('scale_drift', {'cols': ['f0'], 'mean': 0.9, 'sd': 0.01}),
        ('scale_drift', {'cols': ['f0'], 'mean': 1.1, 'sd': 0.01}),
        ('scale_drift', {'cols': ['f0'], 'mean': 1.3, 'sd': 0.01}),
        ('scale_drift', {'cols': ['f0'], 'mean': 1.5, 'sd': 0.01}),
        ('scale_drift', {'cols': ['f0'], 'mean': 2.0, 'sd': 0.01}),
        ('scale_drift', {'cols': ['f0'], 'mean': 5.0, 'sd': 0.01}),
        ('scale_drift', {'cols': ['f0'], 'mean': 10.0, 'sd': 0.01}),
        ('scale_drift', {'cols': ['f0'], 'mean': 20.0, 'sd': 0.01})
    ]

    # Use Performance Predictor
    performance_predictor = PerformancePredictor(model, metric_fn=accuracy_score, corruptions=corruptions, store_train_data=True)
    performance_predictor.fit(X=X_test, y=y_test)
    assert len(performance_predictor.y_train_regressor) == len(corruptions)

    predicted_acc_1 = performance_predictor.predict(X_serving_1)[0]
    real_acc_1 = accuracy_score(y_serving_1, model.predict(X_serving_1))
    assert np.abs(predicted_acc_1 - real_acc_1) < 0.08

    predicted_acc_2 = performance_predictor.predict(X_serving_2)[0]
    real_acc_2 = accuracy_score(y_serving_2, model.predict(X_serving_2))
    assert np.abs(predicted_acc_2 - real_acc_2) < 0.08

    # Performance drop is predicted
    assert predicted_acc_2 < predicted_acc_1

    # Invalid corruptions raises error
    corruptions.append(('invalid_drift_function', {'cols': ['f0'], 'force': 0.0, 'noise': 0.01}))
    with pytest.raises(Exception):
        performance_predictor = PerformancePredictor(model, metric_fn=accuracy_score, corruptions=corruptions, store_train_data=True)
        performance_predictor.fit(X=X_test, y=y_test)


def test_performance_predictor_regression():
    
    # Create Regression Dataset
    X, y = make_regression(1000, n_features=2, n_informative=2)
    X = pd.DataFrame(X, columns=["f" + str(i) for i in range(X.shape[1])])
    # Introduce Random categorical variable for testing purposes
    X["f2"] = np.random.choice([0, 1, 2], size=len(X))
    X_source, X_serving, y_source, y_serving = train_test_split(X, y, test_size=0.5, random_state=42)

    # Split Serving and add Drift
    X_serving_1 = X_serving.iloc[0:250].copy()
    X_serving_2 = X_serving.iloc[250:].copy()
    y_serving_1 = y_serving[0:250]
    y_serving_2 = y_serving[250:]

    feature = "f0"
    p = 0.75
    amount = 100.5
    noise = 0.05
    feature_values = X_serving_2[feature].values
    idx_corrupt = np.random.choice(range(len(feature_values)), size=int(len(feature_values) * p), replace=False)
    shift_values = np.random.normal(loc=amount, scale=noise, size=len(idx_corrupt))
    feature_values[idx_corrupt] += shift_values
    X_serving_2[feature] = feature_values
    # Change distribution in feature 3
    X_serving_2["f2"] = np.random.choice([0, 1, 2], size=len(X_serving_2), p=[0.7, 0.15, 0.15])

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X_source, y_source, test_size=0.3, random_state=42)
    model = LinearRegression()
    model = model.fit(X_train, y_train)

    # Use Performance Predictor - in this case,corruptions are autogenerated
    performance_predictor = PerformancePredictor(
        model, metric_fn=mean_absolute_error, 
        corruptions=None,
        performance_predictor = RandomForestRegressor(criterion='mae', random_state=42)
    )

    dataset_schema = DataSchema().generate_manual(dataframe=X_train, categ_columns=["f2"], discrete_columns=[], binary_columns=[])
    performance_predictor.fit(X=X_test, y=y_test, X_serving=X_serving_1, dataset_schema=dataset_schema)
    predicted_mae_1 = performance_predictor.predict(X_serving_1)[0]
    real_mae_1 = mean_absolute_error(y_serving_1, model.predict(X_serving_1))

    performance_predictor.fit(X=X_test, y=y_test, X_serving=X_serving_2)
    predicted_mae_2 = performance_predictor.predict(X_serving_2)[0]
    real_mae_2 = mean_absolute_error(y_serving_2, model.predict(X_serving_2))

    # Performance Drop is Predicted
    assert (predicted_mae_2 > predicted_mae_1) and (real_mae_2 > real_mae_1)

def test_performance_predictor_multiclass_classification():

    # Create Multiclass Classification Dataset
    X, y = make_classification(n_samples=1000, n_features=3, n_informative=3, n_redundant=0, n_classes=3, random_state=15)
    X = pd.DataFrame(X, columns=["f" + str(i) for i in range(X.shape[1])])
    X_source, X_serving, y_source, y_serving = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Split Serving and add Drift
    X_serving_1 = X_serving.iloc[0:250].copy()
    X_serving_2 = X_serving.iloc[250:].copy()
    y_serving_1 = y_serving[0:250]
    y_serving_2 = y_serving[250:]

    feature = "f0"
    p = 0.5
    amount = 10.5
    noise = 0.05
    feature_values = X_serving_2[feature].values
    idx_corrupt = np.random.choice(range(len(feature_values)), size=int(len(feature_values) * p), replace=False)
    shift_values = np.random.normal(loc=amount, scale=noise, size=len(idx_corrupt))
    feature_values[idx_corrupt] += shift_values
    X_serving_2[feature] = feature_values

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X_source, y_source, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model = model.fit(X_train, y_train)

    # Use Performance Predictor - in this case,corruptions are autogenerated
    performance_predictor = PerformancePredictor(model, metric_fn=accuracy_score, corruptions=None)

    performance_predictor.fit(X=X_test, y=y_test, X_serving=X_serving_1)
    predicted_acc_1 = performance_predictor.predict(X_serving_1)[0]
    real_acc_1 = accuracy_score(y_serving_1, model.predict(X_serving_1))

    performance_predictor.fit(X=X_test, y=y_test, X_serving=X_serving_2)
    predicted_acc_2 = performance_predictor.predict(X_serving_2)[0]
    real_acc_2 = accuracy_score(y_serving_2, model.predict(X_serving_2))

    # Performance Drop is Predicted
    assert (predicted_acc_2 < predicted_acc_1) and (real_acc_2 < real_acc_1)
