"""Validate the public data boundary with independently generated inputs."""
import pytest

from release_support import synthetic_frame, validate_frame


def test_synthetic_schema():
    frame = validate_frame(synthetic_frame())
    assert len(frame) == 720
    assert frame.class_id.nunique() == 3


def test_duplicate_depth_is_rejected():
    frame = synthetic_frame()
    frame.loc[1, 'depth'] = frame.loc[0, 'depth']
    with pytest.raises(ValueError, match='unique'):
        validate_frame(frame)


def test_incorrect_sampling_is_rejected():
    frame = synthetic_frame()
    frame.loc[0, 'depth'] += 0.01
    with pytest.raises(ValueError, match='0.125'):
        validate_frame(frame)


def test_reused_interval_is_rejected():
    frame = synthetic_frame()
    frame.loc[frame.interval_id.eq('ARTIFICIAL_03'), 'interval_id'] = 'ARTIFICIAL_00'
    with pytest.raises(ValueError, match='disconnected'):
        validate_frame(frame)


def test_path_in_well_id_is_rejected():
    frame = synthetic_frame()
    frame['well_id'] = '../escape'
    with pytest.raises(ValueError, match='well ID'):
        validate_frame(frame)


def test_missing_curve_and_nonpositive_resistivity_are_rejected():
    frame = synthetic_frame()
    frame['DT'] = float('nan')
    with pytest.raises(ValueError, match='entire curve'):
        validate_frame(frame)
    frame = synthetic_frame()
    frame.loc[0, 'LLD'] = 0
    with pytest.raises(ValueError, match='positive'):
        validate_frame(frame)
