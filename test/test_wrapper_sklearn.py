import torch
import sklearn.preprocessing

import pytorch_utils.wrapper.sklearn as sklearn_wrap


def test_sklearn_wrapper():
    cdt = sklearn_wrap.torchify(sklearn.preprocessing.LabelEncoder())

    cdt.fit(torch.arange(100, dtype=torch.long) + 100)

    out = cdt.inverse_transform(torch.randint(10, size=(100, )))
    assert isinstance(out, torch.Tensor)

    out = cdt.fit_transform(torch.randint(10, size=(100, )) + 100)
    assert isinstance(out, torch.Tensor)

    out = cdt.transform(torch.randint(10, size=(100, )) + 100)
    assert isinstance(out, torch.Tensor)
