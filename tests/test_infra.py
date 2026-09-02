import numpy as np

from dataloader import MRMultiPlaneDataset, TASKS


def test_mrnet_fixture_loads_with_dataset(mrnet_fixture):
    dataset = MRMultiPlaneDataset(str(mrnet_fixture), train=True)
    assert len(dataset) == 6

    volumes, label, _weights, exam_id = dataset[0]
    assert len(volumes) == 3
    assert volumes[0].dim() == 3 and volumes[0].shape[1:] == (256, 256)
    assert label.shape == (len(TASKS),)
    assert exam_id == "0000"


def test_mrnet_fixture_labels_have_both_classes(mrnet_fixture):
    dataset = MRMultiPlaneDataset(str(mrnet_fixture), train=True)
    labels = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])
    for column in range(labels.shape[1]):
        assert set(labels[:, column].tolist()) == {0.0, 1.0}
