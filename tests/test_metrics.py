import pytest
np = pytest.importorskip("numpy")
from evaluation.metrics import compute_metrics, per_sample_metrics, with_mean_k_aliases


def test_metric_definitions_and_different_minima():
    gt = np.zeros((1, 2, 2))
    predictions = np.array([[[[0., 0.], [2., 0.]], [[4., 0.], [0., 0.]]]])
    m = compute_metrics(predictions, gt)
    assert m == {"ADE": 1.5, "FDE": 1., "minADE@K": 1., "minFDE@K": 0.}
    px = compute_metrics(predictions, gt, np.array([[[10, 20], [30, 40]]]))
    assert px["minADE@K_px"] == 20
    assert px["ADE_px"] == 25
    assert px["FDE_px"] == 30


def test_relative_restore_and_single_sample():
    last = np.array([[.7, .2]])
    future_rel = np.array([[[-.1, .1], [.2, -.1]]])
    truth = future_rel + last[:, None]
    predictions = future_rel[:, None] + last[:, None, None]
    assert all(v == 0 for v in compute_metrics(predictions, truth).values())
    predictions[..., 0] += .1
    m = compute_metrics(predictions, truth)
    assert m["ADE"] == pytest.approx(.1)
    assert m["ADE"] == m["minADE@K"]
    assert m["FDE"] == m["minFDE@K"]


def test_mean_k_names_are_exact_aliases():
    gt = np.zeros((1, 2, 2))
    pred = np.array([[[[0., 0.], [2., 0.]], [[4., 0.], [0., 0.]]]])
    values = per_sample_metrics(pred, gt)
    values.update(per_sample_metrics(pred, gt, np.array([[[10, 20], [30, 40]]])))
    named = with_mean_k_aliases(values, 2)
    for old, new in (("ADE", "meanADE@2"), ("FDE", "meanFDE@2"),
                     ("ADE_px", "meanADE@2_px"), ("FDE_px", "meanFDE@2_px"),
                     ("minADE@K", "minADE@2"), ("minFDE@K", "minFDE@2"),
                     ("minADE@K_px", "minADE@2_px"), ("minFDE@K_px", "minFDE@2_px")):
        np.testing.assert_array_equal(named[old], named[new])
    assert named["minADE@K"][0] == 1.0
    assert named["minFDE@K"][0] == 0.0
