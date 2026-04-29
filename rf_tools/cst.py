from pathlib import Path
import re
import numpy as np
from itertools import islice

from .units import TIME_UNITS, FREQUENCY_UNITS, LENGTH_UNITS

from typing import Literal, Sequence, overload
from .quantities import RealArray, ComplexArray


class CstAsciiParseError(Exception):
    """Exception raised for errors parsing CST ASCII data."""


class CstAsciiUnitParseError(Exception):
    """Exception raised for errors parsing units in CST ASCII data."""


def _parameters_are_close(
    a: dict[str, float], b: dict[str, float],
    rel_tol: float = 1e-09, abs_tol: float = 0.0,
    ensure_same_key_set: bool = True,
) -> bool:
    """Check whether two parameter dictionaries are close in value.

    Args:
        a: First parameter dictionary.
        b: Second parameter dictionary.
        rel_tol: Relative tolerance for float comparison.
        abs_tol: Absolute tolerance for float comparison.
        ensure_same_key_set: If true (default), raises KeyError if `a` and `b` key sets
            are not identical. If false, keys of `b` may be a superset of the keys of `a`.

    Returns:
        True if all shared keys are close according to tolerance.
    """

    if ensure_same_key_set:
        if a.keys() != b.keys():
            raise KeyError('`a` and `b` do not have identical sets of keys')

    for k in a.keys():
        if not np.isclose(a[k], b[k], rtol=rel_tol, atol=abs_tol):
            return False
    return True


@overload
def _get_quantity_from_cst_ascii_lines(
    lines: Sequence[str],
    *,
    is_complex: Literal[True] = ...,
    convert_x_unit: bool = ...,
    header_line_index: int = ...,
) -> tuple[RealArray, ComplexArray]:
    ...


@overload
def _get_quantity_from_cst_ascii_lines(
    lines: Sequence[str],
    *,
    is_complex: Literal[False],
    convert_x_unit: bool = ...,
    header_line_index: int = ...,
) -> tuple[RealArray, RealArray]:
    ...


def _get_quantity_from_cst_ascii_lines(
    lines: Sequence[str],
    is_complex: bool = True,
    convert_x_unit: bool = True,
    header_line_index: int = 1,
) -> tuple[RealArray, RealArray | ComplexArray]:
    
    header_line = lines[header_line_index]

    if convert_x_unit:
        for unit_dict in [TIME_UNITS, FREQUENCY_UNITS, LENGTH_UNITS]:
            unit_match = re.search('(' + '|'.join(unit_dict.keys()) + ')', header_line)
            if unit_match:
                x_unit_factor = unit_dict[unit_match.group(1)]
                break

        else:
            raise CstAsciiUnitParseError(f'Could not determine unit in header line `{header_line}`')
        
    else:
        x_unit_factor = 1

    if is_complex: # complex quantity
        # Try format Real-imaginary
        header_match = re.search(r'(\[Re).*(\[Im)', header_line)
        if header_match:
            raw = np.loadtxt(lines, comments='#')
            return (
                raw[:,0] * x_unit_factor,
                raw[:,1] + 1j * raw[:,2]
            )
        
        # Try format Magnitude-phase
        header_match = re.search(r'(\[Mag).*(\[Pha)', header_line)
        if header_match:
            raw = np.loadtxt(lines, comments='#')
            return (
                raw[:,0] * x_unit_factor,
                raw[:,1] * np.exp(1j * raw[:,2])
            )
    
        raise CstAsciiParseError(f'Could not determine complex format. Is this a complex-valued export from CST?')
    
    else: # real quantity
        raw = np.loadtxt(lines, comments='#')
        return (
            raw[:,0] * x_unit_factor,
            raw[:,1]
        )


@overload
def get_quantity_from_cst_ascii(
    filename: Path | str,
    *,
    parameter_filter: dict[str, float] = ...,
    is_complex: Literal[True] = ...,
    convert_x_unit: bool = ...,
    header_line_index: int = ...,
    silent: bool = ...,
) -> tuple[RealArray, ComplexArray]:
    ...
    

@overload
def get_quantity_from_cst_ascii(
    filename: Path | str,
    *,
    parameter_filter: dict[str, float] = ...,
    is_complex: Literal[False],
    convert_x_unit: bool = ...,
    header_line_index: int = ...,
    silent: bool = ...,
) -> tuple[RealArray, RealArray]:
    ...


def get_quantity_from_cst_ascii(
    filename: Path | str,
    parameter_filter: dict[str, float] = {},
    is_complex: bool = True,
    convert_x_unit: bool = True,
    header_line_index: int = 1,
    silent: bool = True,
) -> tuple[RealArray, RealArray | ComplexArray]:

    # match lines in CST export against this to find parameters
    parameters_pattern = re.compile(r'(\w+)\s*=\s*([-+]?\d+(?:\.\d+)?)')
    no_parameters_pattern = re.compile(r'#$')

    block_slice_start: int | None = None
    block_slice_end: int | None = None
    data_after_header_match = False

    # first, iterate over entire file without actually parsing,
    # this will also ensure that parameter filter is unambiguous
    for n, line in enumerate(open(filename, 'r')):
        
        # check if comment line
        if not line.startswith('#'):
            if block_slice_start is not None:
                data_after_header_match = True
            continue # No processing of data lines

        # block ends when encountering comment after data
        if (block_slice_start is not None) and data_after_header_match:
            block_slice_end = n

        # check if comment line is header row
        if not (
            pairs := parameters_pattern.findall(line)
            or no_parameters_pattern.match(line)
        ):
            continue # not a header row

        if parameter_filter: # user has defined parameters to filter by
            if not _parameters_are_close(
                parameter_filter,
                {k: float(v) for k, v in pairs},
                ensure_same_key_set=False
            ):
                continue # parameters found, but do not match filter

        if block_slice_start is not None:
            raise CstAsciiParseError(f'Parameter filter `{parameter_filter}` is ambiguous, multiple matching blocks found in `{filename}`')
        
        block_slice_start = n
        data_after_header_match = False

    # end of loop, must have a match by now
    if block_slice_start is None:
        raise CstAsciiParseError(f'Could not parse parameter combination `{parameter_filter}` from `{filename}`')

    # actually load content into memory and parse it
    with open(filename, 'r') as fp:
        # for itertools.islice, None means start=0 or end=-1
        block = list(islice(fp, block_slice_start, block_slice_end))
    
    if not silent:
        print(f'Found quantity, block has {len(block)} lines including headers.')
        print(f'  First line: `{block[0][:30]} ...`')
        print(f'  Last line:  `{block[-1][:30]} ...`')
    
    return _get_quantity_from_cst_ascii_lines(
        lines=block,
        is_complex=is_complex,
        convert_x_unit=convert_x_unit,
        header_line_index=header_line_index
    )