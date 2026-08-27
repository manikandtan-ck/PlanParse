import numpy as np

from planparse.evaluation import centerline_chamfer, iou, precision_recall_f1, tolerance_f1


def test_iou():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[2:6, 2:6] = True
    b[4:8, 4:8] = True
    assert round(iou(a, b), 3) == round(4 / 28, 3)


def test_tolerance_f1_accepts_small_shift():
    a = np.zeros((20, 20), bool)
    b = np.zeros((20, 20), bool)
    a[10, 4:16] = True
    b[12, 4:16] = True
    assert tolerance_f1(a, b, 3) > 0.99


def test_metrics_perfect_match():
    gt = np.zeros((20, 20), bool)
    gt[4:12, 5:16] = True
    assert iou(gt, gt) == 1
    assert precision_recall_f1(gt, gt) == (1.0, 1.0, 1.0)
    assert tolerance_f1(gt, gt, 3) == 1
    assert centerline_chamfer(gt, gt) == 0


def test_metrics_no_overlap():
    gt = np.zeros((20, 20), bool)
    pred = np.zeros((20, 20), bool)
    gt[2:5, 2:5] = True
    pred[15:18, 15:18] = True
    assert iou(pred, gt) == 0
    assert precision_recall_f1(pred, gt)[2] == 0
    assert centerline_chamfer(pred, gt) > 0
