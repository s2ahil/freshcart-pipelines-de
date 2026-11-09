
from sklearn.datasets import load_iris

from iris_project.transformers.prepare_data import transform


def test_split_is_80_20():
    # Load the Iris dataset
    df = load_iris(as_frame=True).frame

    # Apply our transformer
    out = transform(df)

    # Verify the 80:20 split
    assert (out["split"] == "train").sum() == 120
    assert (out["split"] == "test").sum() == 30
