import torch
from lightweight_models import FeatBankMRNet
from dataloader import TASKS


def _payload(B=2, S=4, Sm=6, encs=("e1", "e2"), dims=(16, 24), pdim=8, grid=4):
    pooled = {e: {p: torch.randn(B, S, d) for p in ("sagittal", "coronal", "axial")}
              for e, d in zip(encs, dims)}
    patch = {e: {p: torch.randn(B, Sm, pdim, grid, grid) for p in ("sagittal", "coronal")}
             for e in encs}
    return {"pooled": pooled, "patch": patch}


def test_forward_returns_B_by_3():
    m = FeatBankMRNet({"e1": 16, "e2": 24}, {"e1": 8, "e2": 8}, d_model=32)
    out = m(_payload())
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()
    assert TASKS == ("abnormal", "acl", "meniscus")     # order guard


def test_all_params_train_and_receive_grad():
    m = FeatBankMRNet({"e1": 16}, {"e1": 8}, d_model=16)
    out = m(_payload(encs=("e1",), dims=(16,)))
    out.sum().backward()
    assert all(p.requires_grad for p in m.parameters())
    assert all(p.grad is not None for p in m.parameters())


def test_meniscus_head_uses_patch_pyramid():
    m = FeatBankMRNet({"e1": 16}, {"e1": 8}, d_model=16)
    p = _payload(encs=("e1",), dims=(16,))
    base = m(p)[:, 2].clone()
    p["patch"]["e1"]["sagittal"] += 5.0                 # perturb only the pyramid input
    assert not torch.allclose(base, m(p)[:, 2])


def test_works_without_patch_when_no_patch_encoders():
    m = FeatBankMRNet({"e1": 16}, {}, d_model=16, want_patch_encoders=())
    p = _payload(encs=("e1",), dims=(16,))
    p["patch"] = {}
    assert m(p).shape == (2, 3)


def test_empty_want_patch_list_disables_pyramid_and_forward_ignores_patch():
    m = FeatBankMRNet({"e1": 16, "e2": 24}, {"e1": 8, "e2": 8}, d_model=16,
                      want_patch_encoders=[])
    assert m.meniscus_pyramid is None
    p = _payload(dims=(16, 24))
    p["patch"] = {}
    out = m(p)
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()
