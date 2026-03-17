import os, shutil

from mercury.monitoring.create_tutorials import create_tutorials


def test_create_tutorials():
    # Silently remove the full tree './monitoring_tutorials/'
    shutil.rmtree('./monitoring_tutorials', ignore_errors = True)

    create_tutorials('./')

    assert os.path.isfile('./monitoring_tutorials/drift/Autoencoder_Drift_Detection.ipynb')

    # Clean up
    shutil.rmtree('./monitoring_tutorials', ignore_errors = True)


if __name__ == "__main__":
    test_create_tutorials()
