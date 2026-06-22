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

    # regular expressions to find header lines (no matter if with parameters or not)
    header_line_pattern = re.compile(r'(^#Parameters|^#$)')
    # regular expression to find parameter-value pairs
    parameters_pattern = re.compile(r'(\w+)\s*=\s*([-+]?\d+(?:\.\d+)?)')

    # first, iterate over entire file without actually parsing,
    # this will also ensure that parameter filter is unambiguous

    # collect header lines and their indices
    header_lines: list[tuple[int, str]] = []
    for line_number, line in enumerate(open(filename, 'r')):
        if not line.startswith('#'):
            continue # not a comment line
        if not header_line_pattern.match(line):
            continue # not a header line
        header_lines.append((line_number, line))

    block_slice_start: int | None = None
    block_slice_end: int | None = None

    if not parameter_filter:
        # if user specifies no parameter filter, there must be exactly one match
        if not len(header_lines) == 1:
            raise CstAsciiParseError(f'No parameter filter specified, but {len(header_lines)} blocks found in `{filename}`') 
        # for islice, `None` indicated full range
        block_slice_start = None
        block_slice_end = None
    else:
        # if user specified parameter filter, there must be exactly one match
        for n, (line_number, line) in enumerate(header_lines):
            if _parameters_are_close(
                parameter_filter,
                {k: float(v) for k, v in parameters_pattern.findall(line)},
                ensure_same_key_set=False
            ):
                if block_slice_start is not None:
                    # already matched before
                    raise CstAsciiParseError(f'Parameter filter `{parameter_filter}` is ambiguous, multiple matching headerlines found (at least) in line {block_slice_start} and line {line_number} in `{filename}`')
                block_slice_start = line_number
                try:
                    block_slice_end = header_lines[n+1][0]
                except IndexError: # last block has no next header
                    block_slice_end = None

        # raise error if no match at all
        if block_slice_start is None:
            raise CstAsciiParseError(f'Parameter filter `{parameter_filter}` delivered no matches in `{filename}`')
                

    # actually load content into memory and parse it
    with open(filename, 'r') as fp:
        # for itertools.islice, `None` means full range
        block = list(islice(fp, block_slice_start, block_slice_end))
    
    if not silent:
        print(f'Found quantity, block has {len(block)} lines including headers.')
        print(f'  First line: `{block[0][:50]} ...`')
        print(f'  Last line:  `{block[-1][:50]} ...`')
    
    return _get_quantity_from_cst_ascii_lines(
        lines=block,
        is_complex=is_complex,
        convert_x_unit=convert_x_unit,
        header_line_index=header_line_index
    )
