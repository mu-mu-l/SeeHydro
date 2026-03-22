"""归一化模块单元测试."""

import numpy as np
import pytest

from seehydro.preprocessing.normalize import (
    normalize_image,
    normalize_minmax,
    normalize_percentile,
)


def _make_test_array(shape, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=10.0, scale=5.0, size=shape).astype(np.float32)


def _assert_normalized_output(y, expected_shape):
    assert isinstance(y, np.ndarray)
    assert y.shape == expected_shape
    assert y.dtype == np.float32
    assert np.isfinite(y).all()
    assert y.min() >= -1e-6
    assert y.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("shape", [(32, 48), (6, 32, 48)])
def test_normalize_percentile_输出范围和类型正确(shape):
    x = _make_test_array(shape)
    y = normalize_percentile(x)
    _assert_normalized_output(y, shape)


@pytest.mark.parametrize("shape", [(32, 48), (6, 32, 48)])
def test_normalize_minmax_输出范围和类型正确(shape):
    x = _make_test_array(shape)
    y = normalize_minmax(x)
    _assert_normalized_output(y, shape)


@pytest.mark.parametrize("shape", [(32, 48), (6, 32, 48)])
def test_normalize_image_默认方法_输出正确(shape):
    x = _make_test_array(shape)
    y = normalize_image(x)
    _assert_normalized_output(y, shape)


@pytest.mark.parametrize("shape", [(16, 16), (4, 16, 16)])
@pytest.mark.parametrize("normalizer", [normalize_percentile, normalize_minmax])
def test_全零输入_不报错(shape, normalizer):
    x = np.zeros(shape, dtype=np.float32)
    y = normalizer(x)
    _assert_normalized_output(y, shape)
