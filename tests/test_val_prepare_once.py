import inspect

import train


def test_val_path_calls_prepare_inputs_once_per_batch():
    """In eval, prepare_inputs must run once (inside forward_with_eval_policy),
    not twice (it used to also run unconditionally at the top of the loop).

    Asserted structurally on the source of iterate_epoch: wiring a full
    single-batch iterate_epoch call is heavy relative to the signal, so this
    pins the guard placement instead. See final-fix-report.md for the tradeoff.
    """
    src = inspect.getsource(train.iterate_epoch)
    # the unconditional top-of-loop call must be gone
    assert "axial = prepare_inputs(volumes, device, args)\n\n        if is_train" not in src
    # and the call must now be guarded
    assert "if is_train:\n            sagittal, coronal, axial = prepare_inputs" in src
