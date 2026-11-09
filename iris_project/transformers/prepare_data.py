try:
    from mage_ai.data_preparation.decorators import transformer
except ImportError:
    def transformer(func):
        return func

from sklearn.model_selection import train_test_split


@transformer
def transform(df, *args, **kwargs):
    """Add an 80:20 train/test split to the Iris dataset."""

    output = df.copy()

    train_indices, test_indices = train_test_split(
        output.index,
        test_size=0.2,
        random_state=42,
        stratify=output["target"]
    )

    output["split"] = "train"
    output.loc[test_indices, "split"] = "test"

    return output
