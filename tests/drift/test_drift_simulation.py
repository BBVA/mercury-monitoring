import pytest

import seaborn as sns
import numpy as np

from mercury.monitoring.drift.drift_simulation import BatchDriftGenerator


@pytest.fixture(scope='module')
def datasets():
    tips = sns.load_dataset('tips')
    tips['sex'] = tips['sex'].astype(str)
    tips['smoker'] = tips['smoker'].astype(str)
    tips['day'] = tips['day'].astype(str)
    tips['time'] = tips['time'].astype(str)

    titanic = sns.load_dataset('titanic')
    isna_deck = titanic.deck.isna()
    titanic['class'] = titanic['class'].astype(str)
    titanic['deck'] = titanic['deck'].astype(str)
    titanic['who'] = titanic['who'].astype(str)
    titanic.loc[isna_deck, 'deck'] = np.nan

    return tips, titanic


def test_constructor(datasets):
    tips, titanic = datasets
    data_array = np.array(titanic)
    # Test this can be built with a np.array
    sim = BatchDriftGenerator(data_array)


def test_na_drift(datasets):
    tips, titanic = datasets

    sim = (BatchDriftGenerator(titanic.copy())
           .missing_val_drift(cols=['alive'], percent=1))

    assert sim.data.alive.isna().sum() == len(sim.data)

    sim = (BatchDriftGenerator(titanic.copy())
           .missing_val_drift(cols=['alive'], percent=0))

    assert sim.data.alive.isna().sum() == titanic.alive.isna().sum()

    sim = (BatchDriftGenerator(titanic.copy())
           .missing_val_drift(cols=['alive'], percent=.5))

    assert (sim.data.alive.isna().sum() / len(sim.data)) == pytest.approx(0.5, 0.2)


def test_shift_drift(datasets):
    tips, titanic = datasets

    # This will only adds noise to continuous cols
    sim = BatchDriftGenerator(titanic.copy()).shift_drift(force=10)

    assert sim.data.age.mean() > titanic.age.mean()
    assert sim.data.fare.mean() > titanic.fare.mean()
    # This is  binary
    assert sim.data.survived.mean() == titanic.survived.mean()


def test_scale_drift(datasets):
    tips, titanic = datasets

    # This will only adds noise to continuous cols
    sim = BatchDriftGenerator(titanic.copy()).scale_drift(iqr=[1.1, 1.3])

    assert sim.data.age.mean() > titanic.age.mean()
    assert sim.data.fare.mean() > titanic.fare.mean()
    # This is  binary
    assert sim.data.survived.mean() == titanic.survived.mean()

    # This will only adds noise to continuous cols
    sim = BatchDriftGenerator(titanic.copy()).scale_drift()		# 5% unbiased noise by default
    # This is  binary
    assert sim.data.survived.mean() == titanic.survived.mean()


def test_recodification_drift(datasets):
    tips, titanic = datasets

    sim = BatchDriftGenerator(titanic.copy()).recodification_drift(cols=['deck'])

    assert titanic.deck.isna().sum() == sim.data.deck.isna().sum()

    # All counts all equal
    assert (sim.data.deck.value_counts().values == titanic.deck.value_counts().values).all()

    # Ids are flipped
    assert not (sim.data.deck.value_counts().index == titanic.deck.value_counts().index).all()

    sim = BatchDriftGenerator(titanic.copy()).recodification_drift()
    assert titanic.deck.isna().sum() == sim.data.deck.isna().sum()
    assert (sim.data.deck.value_counts().values == titanic.deck.value_counts().values).all()
    assert not (sim.data.deck.value_counts().index == titanic.deck.value_counts().index).all()
    assert (titanic.fare == sim.data.fare).all()


def test_rotation_drift(datasets):
    tips, titanic = datasets

    sim = (BatchDriftGenerator(titanic.copy())
           .hyperplane_rotation_drift(force=0))

    assert (titanic.fare == sim.data.fare).all()
    # assert (titanic.age == sim.data.age).all()

    # Assert applying 180 deg transformation mirrors things
    sim = sim.hyperplane_rotation_drift(force=180)
    assert not (titanic.dropna().fare == sim.data.dropna().fare).all()

    # Assert 180 deg again returns it to the start
    sim = sim.hyperplane_rotation_drift(force=180)

    # Its actually the same but due to data type changes a hard comparison with "==" may fail
    assert np.allclose(titanic.fare, sim.data.fare)

    # Ensure distances to origin before and after rotations are the same
    assert (
            np.linalg.norm(titanic.loc[:, ['age', 'fare']].dropna()) ==  # noqa: E126, E502, W504
            pytest.approx(np.linalg.norm(sim.data.loc[:, ['age', 'fare']].dropna()))
    )

def test_generate_outliers_drift_percentile_method(datasets):
    tips, titanic = datasets

    # One column modified
    method_params = {"percentile": 99, "proportion_outliers": 1}
    sim = BatchDriftGenerator(titanic.copy()).outliers_drift(
        cols=["age"], method="percentile", method_params=method_params
    )
    perc_99 = np.percentile(titanic["age"].dropna(), 99)
    prop_outliers = (sim.data["age"] <= perc_99).sum() / len(sim.data)
    assert prop_outliers == pytest.approx(1, 0.02)
    assert sim.data["age"].mean() > titanic["age"].mean()

    # Two columns modified
    method_params = {"percentile": 1, "proportion_outliers": 0.75}
    sim = BatchDriftGenerator(titanic.copy()).outliers_drift(
        cols=["age", "fare"], method="percentile", method_params=method_params
    )
    for col in ["age", "fare"]:
        perc_1 = np.percentile(titanic[col].dropna(), 1)
        prop_outliers = (sim.data[col] <= perc_1).sum() / len(sim.data)
        assert prop_outliers == pytest.approx(0.75, 0.02)
        assert sim.data[col].mean() < titanic[col].mean()


def test_generate_outliers_drift_set_value_method(datasets):
    tips, titanic = datasets

    method_params = {"value": 90., "proportion_outliers": 0.95}
    sim = BatchDriftGenerator(titanic.copy()).outliers_drift(
        cols=["age"], method="value", method_params=method_params
    )
    prop_outliers = (sim.data["age"] == 90.0).sum() / len(sim.data)
    assert prop_outliers == pytest.approx(0.95, 0.02)
    assert sim.data["age"].mean() > titanic["age"].mean()

def test_generate_outliers_drift_custom_fn(datasets):
    tips, titanic = datasets

    def my_outlier_gen_fn(X, params=dict):
        _params = dict(
            multiplier=3,
            proportion_outliers=0.5
        )
        if isinstance(params, dict):
            _params.update(params)

        outlier_val = X[~np.isnan(X)].max() * _params["multiplier"]
        indices = np.random.choice(range(len(X)), size=int(len(X) * _params["proportion_outliers"]), replace=False)
        X[indices] = outlier_val
        return X

    method_params = {"multiplier": 5, "proportion_outliers": 0.9}
    sim = BatchDriftGenerator(titanic.copy()).outliers_drift(
        cols=["age"], method=my_outlier_gen_fn, method_params=method_params
    )
    prop_outliers = (sim.data["age"] > 100.0).sum() / len(sim.data)
    assert prop_outliers == pytest.approx(0.90, 0.02)
    assert sim.data["age"].mean() > titanic["age"].mean()

def test_generate_outliers_discrete_column(datasets):
    tips, titanic = datasets
    titanic["fare"] = titanic["fare"].astype(int)

    method_params = {"percentile": 99, "proportion_outliers": 1}
    sim = BatchDriftGenerator(titanic.copy()).outliers_drift(
        cols=["fare"], method="percentile", method_params=method_params
    )

    perc_99 = np.percentile(titanic["fare"].dropna(), 99)
    prop_outliers = (sim.data["fare"] <= perc_99).sum() / len(sim.data)
    assert prop_outliers == pytest.approx(1, 0.02)
    assert sim.data["fare"].mean() > titanic["fare"].mean()

def test_generate_outliers_invalid_inputs(datasets):
    tips, titanic = datasets

    # wrong column for percentile method
    method_params = {"percentile": 99, "proportion_outliers": 1}
    with pytest.raises(ValueError):
        sim = BatchDriftGenerator(titanic.copy()).outliers_drift(
            cols=["who"], method="percentile", method_params=method_params
        )

    # wrong specifeid method
    method_params = {"percentile": 99, "proportion_outliers": 1}
    with pytest.raises(ValueError):
        sim = BatchDriftGenerator(titanic.copy()).outliers_drift(
            cols=["fare"], method="wrong_method", method_params=method_params
        )