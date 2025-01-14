import nnsb as avl
import numpy as np

from tests.utils import create_localization_pipeline, queries


def test_recall():
    """
    Validates the metric using a reasonable level of threshold.
    Since one image was taken outside the test map,
    the recall value should be equal to 0.5
    """
    localization_pipeline = create_localization_pipeline()
    recall = avl.get_recall(
        queries, localization_pipeline, k_closest=2
    )

    assert np.isclose(recall, 1)


def test_recall_low_threshold():
    """
    Validates the metric using a low level of threshold.
    System can't locate queries with such accuracy,
    so the recall value should be equal to 0
    """
    localization_pipeline = create_localization_pipeline()
    recall = avl.get_recall(
        queries, localization_pipeline, k_closest=1
    )

    assert np.isclose(recall, 1)
