
if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def validate_iris(df, *args, **kwargs):
    # Check for missing values
    missing_values = df.isnull().sum().sum()

    # Count unique Iris classes
    class_count = df['target'].nunique()

    print("Missing values:", missing_values)
    print("Number of Iris classes:", class_count)

    assert missing_values == 0, "Dataset contains missing values"
    assert class_count == 3, "Dataset must contain exactly 3 classes"

    print("Iris data validation successful!")
    return df


@test
def test_validation(output, *args):
    assert output.isnull().sum().sum() == 0
    assert output['target'].nunique() == 3
