import torch

from temis.model import align_to_options, consistency_loss


def test_align_to_options_inverts_shuffle():
    p = torch.tensor([[0.2, 0.5, 0.3, 0.0]])
    assert torch.allclose(align_to_options(p, [[2, 0, 1]], [3]), torch.tensor([[0.5, 0.3, 0.2, 0.0]]))


def test_consistency_loss_zero_when_same_distribution_in_option_order():
    la = torch.tensor([[1.0, 2.0, 3.0, -1e4]])
    lb = torch.tensor([[3.0, 1.0, 2.0, -1e4]])
    m = torch.tensor([[True, True, True, False]])
    assert consistency_loss(la, lb, m, m, [[0, 1, 2]], [[2, 0, 1]], [3]).item() < 1e-6
    assert consistency_loss(la, la, m, m, [[0, 1, 2]], [[2, 0, 1]], [3]).item() > 0.1


def test_consistency_loss_score_same_order():
    la = torch.tensor([[1.0, 2.0, 3.0]])
    lb = torch.tensor([[1.1, 2.0, 2.9]])
    m = torch.ones(1, 3, dtype=torch.bool)
    assert 0 < consistency_loss(la, lb, m, m, [[0, 1, 2]], [[0, 1, 2]], [3]).item() < 0.05
