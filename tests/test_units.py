import pytest

from rf_tools.units import (
    FREQ_UNITS, LENGTH_UNITS, PREFIXES, TIME_UNITS, format_quantity
)


@pytest.mark.parametrize('units, suffix', [
    (TIME_UNITS, 's'),
    (FREQ_UNITS, 'Hz'),
    (LENGTH_UNITS, 'm'),
])
def test_unit_tables_follow_prefixes(units, suffix):
    assert units == {prefix + suffix: value for prefix, value in PREFIXES.items()}


def test_unit_table_spot_values():
    assert TIME_UNITS['ns'] == 1e-9
    assert FREQ_UNITS['GHz'] == 1e9
    assert LENGTH_UNITS['mm'] == 1e-3


@pytest.mark.parametrize('value, expected', [
    (1, '1.000 Hz'),
    (150, '150.000 Hz'),
    (1234, '1.234 kHz'),
    (-3e9, '-3.000 GHz'),
    (0.5, '500.000 mHz'),
    (2e-4, '200.000 uHz'),
    (1e15, '1.000e+15 Hz'),
    (1e-16, '1.000e-16 Hz'),
])
def test_format_quantity(value, expected):
    assert format_quantity(value, 'Hz') == expected
