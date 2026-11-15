
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def evaluate(df, *args, **kwargs):
    y_true = df['target']
    y_pred = df['predicted']

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Calculate classification metrics
    metrics = pd.DataFrame([{
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        ),
        'Recall': recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        ),
        'F1 Score': f1_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
    }])

    print("\nEvaluation Metrics:")
    print(metrics.to_string(index=False))

    return metrics


@test
def test_evaluation(output, *args):
    assert not output.empty, "Metrics table is empty"
    assert output['Accuracy'].between(0, 1).all()
    assert output['F1 Score'].between(0, 1).all()
