# mercury-monitoring

[![](https://github.com/BBVA/mercury-monitoring/actions/workflows/test.yml/badge.svg)](https://github.com/BBVA/mercury-monitoring)
![](https://img.shields.io/badge/latest-1.1.3-blue)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3816/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3128/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3131/)
[![Apache 2 license](https://shields.io/badge/license-Apache%202-blue)](http://www.apache.org/licenses/LICENSE-2.0)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/BBVA/mercury-monitoring/issues)

***mercury-monitoring*** is a library to monitor data and model drift. It contains a set of utilities which you can use for detecting in advance whether the statistical properties of your data have changed and whether your model's performance is getting worse.

Among others, you can check things like:

- Changes in your entire data distribution (P(X))
- Changes in your marginals (i.e. individial features P(Xi))
- Estimated performance of your model when you don't have access to the true labels

and take the actions you consider pertinent (e.g. retrain your model on more recent data, generate alerts, etc.).

Most of the classes for detecting drift have a `calculate_drift()` method which will yield the predicted result on whether drift has been detected or not.

### Usage example
```python
from mercury.monitoring.drift.domain_classifier_drift import DomainClassifierDrift

# Create example datasets
X_src = np.array([np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)]).T
X_target = np.array([np.random.normal(0, 1, 500) + 2, np.random.normal(0, 1, 500)]).T

# Calculate drift
drift_detector = DomainClassifierDrift(
    X_src,                    # Source dataset
    X_target,                 # Target dataset (e.g. newer batch of data)
    features=["f1", "f2"],
    p_val=0.01,
    test_size=0.3,
    n_runs=10
)

# Get results
drift_metrics = drift_detector.calculate_drift()

# Print the results
print("Drift Score: ", drift_metrics["score"])
print("p_val: ", drift_metrics["p_val"])
print("Is drift detected? ", drift_metrics["drift_detected"])

# Plot drift scores per feature
drift_detector.plot_feature_drift_scores(figsize=(8,4))
```

Feel free to explore more of the drift detectors and use the most appropiate for your use case.

### Documentation
We encourage you checking the documentation of this package at: https://bbva.github.io/mercury-monitoring/site/ and taking a look at the available [notebook tutorials](https://github.com/BBVA/mercury-monitoring/tree/readme/tutorials).

## Mercury project at BBVA

Mercury is a collaborative library that was developed by the Advanced Analytics community at BBVA. Originally, it was created as an [InnerSource](https://en.wikipedia.org/wiki/Inner_source) project but after some time, we decided to release certain parts of the project as Open Source.
That's the case with the `mercury-monitoring` package.

If you're interested in learning more about the Mercury project, we recommend reading this blog [post](https://www.bbvaaifactory.com/mercury-acelerando-la-reutilizacion-en-ciencia-de-datos-dentro-de-bbva/) from www.bbvaaifactory.com

## User installation

The easiest way to install `mercury-monitoring` is using ``pip``:

    pip install -U mercury-monitoring

## Help and support

This library is currently maintained by a dedicated team of data scientists and machine learning engineers from BBVA.  In case you find any bug, have an idea or request feel free to open a ticket on the [Issues](https://github.com/BBVA/mercury-monitoring/issues) section.

### Email
mercury.group@bbva.com
