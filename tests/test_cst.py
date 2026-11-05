from pathlib import Path

import numpy as np
import pytest

from rf_tools.cst import (
    CstAsciiParseError, CstAsciiUnitParseError, read_cst_ascii
)

REAL_IMAG_HEADER = '#"Frequency / GHz"\t"Z [Re] / Ohm"\t"Z [Im] / Ohm"\n'
MAG_PHASE_HEADER = '#"Frequency / GHz"\t"Z [Magnitude] / Ohm"\t"Z [Phase] / rad"\n'
REAL_HEADER = '#"Time / ns"\t"Wake / V/pC"\n'
SEPARATOR = '#' + '-' * 60 + '\n'


def block(header: str, rows: list[list[float]], parameters: str | None = None) -> str:
    lines = []
    if parameters is not None:
        lines.append(f'#Parameters = {{{parameters}}}\n')
    lines += [header, SEPARATOR]
    lines += ['\t'.join(f'{v:.6g}' for v in row) + '\n' for row in rows]
    return ''.join(lines)


def write(tmp_path: Path, *blocks: str) -> Path:
    path = tmp_path / 'export.txt'
    path.write_text(''.join(blocks))
    return path


ROWS = [[0.0, 1.0, 0.5], [1.0, 2.0, -0.5], [2.0, 3.0, 0.0]]
WAKE_ROWS = [[0.0, 1.0], [0.5, -2.0], [1.0, 0.25]]


def test_real_single_block(tmp_path):
    path = write(tmp_path, block(REAL_HEADER, WAKE_ROWS))
    x, y = read_cst_ascii(path, is_complex=False)
    np.testing.assert_allclose(x, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(y, [1.0, -2.0, 0.25])


def test_complex_real_imag(tmp_path):
    path = write(tmp_path, block(REAL_IMAG_HEADER, ROWS))
    x, y = read_cst_ascii(path, is_complex=True)
    np.testing.assert_allclose(x, [0, 1, 2])
    np.testing.assert_allclose(y, [1 + 0.5j, 2 - 0.5j, 3])


def test_complex_mag_phase(tmp_path):
    path = write(tmp_path, block(MAG_PHASE_HEADER, ROWS))
    _, y = read_cst_ascii(path, is_complex=True)
    np.testing.assert_allclose(np.abs(y), [1, 2, 3])
    np.testing.assert_allclose(np.angle(y), [0.5, -0.5, 0])


def test_complex_requested_on_real_export(tmp_path):
    path = write(tmp_path, block(REAL_HEADER, WAKE_ROWS))
    with pytest.raises(CstAsciiParseError):
        read_cst_ascii(path, is_complex=True)


@pytest.mark.parametrize('x_multiplier, y_multiplier, x_factor, y_factor', [
    ('G', 'm', 1e9, 1e-3),
    (1e9, 2.0, 1e9, 2.0),
    ('', 1, 1, 1),
])
def test_multipliers(tmp_path, x_multiplier, y_multiplier, x_factor, y_factor):
    path = write(tmp_path, block(REAL_HEADER, WAKE_ROWS))
    x, y = read_cst_ascii(
        path, is_complex=False, x_multiplier=x_multiplier, y_multiplier=y_multiplier
    )
    np.testing.assert_allclose(x, x_factor * np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(y, y_factor * np.array([1.0, -2.0, 0.25]))


@pytest.mark.parametrize('kwargs', [dict(x_multiplier='X'), dict(y_multiplier='kHz')])
def test_unknown_multiplier(tmp_path, kwargs):
    path = write(tmp_path, block(REAL_HEADER, WAKE_ROWS))
    with pytest.raises(CstAsciiUnitParseError):
        read_cst_ascii(path, is_complex=False, **kwargs)


# --- parameter sweeps (multiple blocks) ---

@pytest.fixture
def sweep_file(tmp_path) -> Path:
    return write(
        tmp_path,
        block(REAL_HEADER, [[0, 1], [1, 1]], parameters='length=10; radius=2.5'),
        block(REAL_HEADER, [[0, 2], [1, 2]], parameters='length=10; radius=5'),
        block(REAL_HEADER, [[0, 3], [1, 3]], parameters='length=20; radius=5'),
    )


@pytest.mark.parametrize('parameter_filter, expected', [
    (dict(length=10, radius=2.5), 1),
    (dict(length=10, radius=5), 2),
    (dict(length=20), 3),
    (dict(radius=2.5), 1),
])
def test_parameter_filter_selects_block(sweep_file, parameter_filter, expected):
    _, y = read_cst_ascii(sweep_file, is_complex=False, parameter_filter=parameter_filter)
    np.testing.assert_array_equal(y, [expected, expected])


@pytest.mark.parametrize('parameter_filter', [
    {},
    dict(length=10),
    dict(radius=5),
], ids=['no-filter', 'ambiguous-length', 'ambiguous-radius'])
def test_parameter_filter_ambiguous(sweep_file, parameter_filter):
    with pytest.raises(CstAsciiParseError):
        read_cst_ascii(sweep_file, is_complex=False, parameter_filter=parameter_filter)


@pytest.mark.parametrize('parameter_filter', [
    dict(length=30),
    dict(width=1),
], ids=['no-matching-value', 'unknown-parameter'])
def test_parameter_filter_no_match(sweep_file, parameter_filter):
    with pytest.raises(CstAsciiParseError):
        read_cst_ascii(sweep_file, is_complex=False, parameter_filter=parameter_filter)


def test_parameter_filter_negative_values(tmp_path):
    path = write(
        tmp_path,
        block(REAL_HEADER, [[0, 1], [1, 1]], parameters='offset=-1.5'),
        block(REAL_HEADER, [[0, 2], [1, 2]], parameters='offset=1.5'),
    )
    _, y = read_cst_ascii(path, is_complex=False, parameter_filter=dict(offset=-1.5))
    np.testing.assert_array_equal(y, [1, 1])


@pytest.mark.parametrize('written, value', [
    ('2', 2.0),
    ('+2', 2.0),
    ('-2', -2.0),
    ('2.5', 2.5),
    ('2.', 2.0),
    ('.5', 0.5),
    ('-.5', -0.5),
    ('2e-3', 2e-3),
    ('2E-3', 2e-3),
    ('2.5e+3', 2.5e3),
    ('-2.5e3', -2.5e3),
    ('.5e-3', 0.5e-3),
    ('2.e3', 2e3),
])
def test_parameter_filter_number_formats(tmp_path, written, value):
    path = write(
        tmp_path,
        block(REAL_HEADER, [[0, 1], [1, 1]], parameters='sigma=999'),
        block(REAL_HEADER, [[0, 2], [1, 2]], parameters=f'sigma={written}'),
    )
    _, y = read_cst_ascii(path, is_complex=False, parameter_filter=dict(sigma=value))
    np.testing.assert_array_equal(y, [2, 2])


@pytest.mark.parametrize('is_complex, header, row', [
    (False, REAL_HEADER, [1.0, 2.0]),
    (True, REAL_IMAG_HEADER, [1.0, 2.0, 3.0]),
], ids=['real', 'complex'])
def test_single_data_row(tmp_path, is_complex, header, row):
    path = write(tmp_path, block(header, [row]))
    x, y = read_cst_ascii(path, is_complex=is_complex)
    np.testing.assert_allclose(x, [1.0])
    assert len(y) == 1


def test_file_without_header(tmp_path):
    path = tmp_path / 'export.txt'
    path.write_text('0\t1\n1\t2\n')
    with pytest.raises(CstAsciiParseError):
        read_cst_ascii(path, is_complex=False)
