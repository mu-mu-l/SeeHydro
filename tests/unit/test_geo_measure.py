"""GIS量测工具单元测试."""

import math

import pytest
from rasterio.transform import from_origin

from seehydro.extraction.geo_measure import (
    geo_to_pixel,
    get_utm_crs,
    measure_distance_m,
    pixel_to_geo,
)


def test_pixel_to_geo_已知变换_坐标正确():
    transform = from_origin(100.0, 40.0, 0.01, 0.01)
    # pixel_to_geo 接受 (col, row) 格式
    lon, lat = pixel_to_geo((20, 10), transform)

    expected_lon = 100.0 + (20 + 0.5) * 0.01
    expected_lat = 40.0 - (10 + 0.5) * 0.01

    assert abs(lon - expected_lon) < 0.01
    assert abs(lat - expected_lat) < 0.01


def test_geo_to_pixel_已知坐标_像素正确():
    transform = from_origin(120.0, 30.0, 0.0001, 0.0001)
    col, row = 200, 100

    lon = 120.0 + (col + 0.5) * 0.0001
    lat = 30.0 - (row + 0.5) * 0.0001

    got_col, got_row = geo_to_pixel(lon, lat, transform)

    assert math.isclose(got_col, col, abs_tol=1)
    assert math.isclose(got_row, row, abs_tol=1)


def test_pixel_geo_往返_一致性():
    transform = from_origin(100.0, 40.0, 0.01, 0.01)
    original_col, original_row = 20, 10

    lon, lat = pixel_to_geo((original_col, original_row), transform)
    got_col, got_row = geo_to_pixel(lon, lat, transform)

    assert math.isclose(got_col, original_col, abs_tol=1)
    assert math.isclose(got_row, original_row, abs_tol=1)


def test_measure_distance_北京到上海_约1000km():
    """北京到上海球面距离约 1060 km."""
    beijing = (116.4074, 39.9042)
    shanghai = (121.4737, 31.2304)

    distance_m = measure_distance_m(beijing, shanghai)

    assert 900_000 <= distance_m <= 1_200_000


@pytest.mark.parametrize("lon,lat,expected_epsg", [
    (116.4074, 39.9042, "EPSG:32650"),  # 北京 UTM Zone 50N
    (121.4737, 31.2304, "EPSG:32651"),  # 上海 UTM Zone 51N
])
def test_get_utm_crs_zone计算正确(lon, lat, expected_epsg):
    crs = get_utm_crs(lon, lat)
    assert crs == expected_epsg
