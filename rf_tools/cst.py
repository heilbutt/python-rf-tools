from pathlib import Path
import re
import numpy as np
from itertools import islice

from .units import TIME_UNITS, FREQUENCY_UNITS, LENGTH_UNITS
from .quantities import RealQuantity, ComplexQuantity

from typing import Literal, Sequence, overload


class CstAsciiParseError(Exception):
    """Exception raised for errors parsing CST ASCII data."""


class CstAsciiUnitParseError(Exception):
    """Exception raised for errors parsing units in CST ASCII data."""


def _is_close(a: float, b: float, rel_tol: float = 1e-09, abs_tol: float = 0.0):
    """Check if two floats are close.
    Between rel_tol and abs_tol, the less strict one takes precedence.

    Args:
        a: First value.
        b: Second value.
        rel_tol: Relative tolerance.
        abs_tol: Absolute tolerance.

    Returns:
        True if the values are close within tolerances.
    """
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


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
        if not _is_close(a[k], b[k], rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


@overload
def _get_quantity_from_cst_ascii_lines(
    lines: Sequence[str],
    *,
    is_complex: Literal[True] = ...,
    convert_x_unit: bool = ...,
    header_line_index: int = ...,
) -> ComplexQuantity:
    ...


@overload
def _get_quantity_from_cst_ascii_lines(
    lines: Sequence[str],
    *,
    is_complex: Literal[False],
    convert_x_unit: bool = ...,
    header_line_index: int = ...,
) -> RealQuantity:
    ...


def _get_quantity_from_cst_ascii_lines(
    lines: Sequence[str],
    is_complex: bool = True,
    convert_x_unit: bool = True,
    header_line_index: int = 1,
) -> RealQuantity | ComplexQuantity:
    
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
            return ComplexQuantity(
                raw[:,0] * x_unit_factor,
                raw[:,1] + 1j * raw[:,2]
            )
        
        # Try format Magnitude-phase
        header_match = re.search(r'(\[Mag).*(\[Pha)', header_line)
        if header_match:
            raw = np.loadtxt(lines, comments='#')
            return ComplexQuantity(
                raw[:,0] * x_unit_factor,
                raw[:,1] * np.exp(1j * raw[:,2])
            )
    
        raise CstAsciiParseError(f'Could not determine complex format. Is this a complex-valued export from CST?')
    
    else: # real quantity
        raw = np.loadtxt(lines, comments='#')
        return RealQuantity(
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
) -> ComplexQuantity:
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
) -> RealQuantity:
    ...


# def get_quantity_from_cst_ascii(
#     filename: Path | str,
#     parameter_filter: dict[str, float] = {},
#     is_complex: bool = True,
#     convert_x_unit: bool = True,
#     header_line_index: int = 1,
#     silent: bool = True,
# ) -> RealQuantity | ComplexQuantity:
#     """Select one trace matching parameters from a CST parametric sweep file. 
#     The ASCII file is obtained by selecting an *parametric* impedance trace in CST and
#     using Post-Processing > Import/Export > ASCII.

#     Args:
#         filename: Path to CST sweep ASCII file.
#         parameter_filter: Matching criteria for parameters. Does not need to include all parameters
#             from the CST project, just enough to unambigously identify the trace.
#         is_complex: Wether the CST quantitiy is real or complex valued.
#         convert_x_unit: If True, will infer X-axis unit from file and convert to base unit.
#         header_line_index: Which line (starting with 0) in a block is the table header.
#         silent: If False, print search status.

#     Returns:
#         Quantity matching the parameter filter. RealQuantity if `is_complex` is false,
#         or ComplexQuantity `is_complex` is true.

#     Raises:
#         CstAsciiParseError: If parsing fails, if there is no match for `parameter_filter`,
#             or if there is *more than one* match for `parameter_filter.`
#     """

#     # match lines in CST export against this to find parameters
#     parameter_pattern = re.compile(r'(\w+)\s*=\s*([-+]?\d+(?:\.\d+)?)')

#     block: list[str] = []
#     block_matches: bool = False
#     any_previous_block_matches: bool = False
#     quantity = None

#     for line in open(filename):

#         if block_matches:
#             block.append(line)

#         # No further processing of data lines
#         if not line.startswith('#'):
#             continue

#         # Line is a comment, try to parse parameters
#         pairs = parameter_pattern.findall(line)
#         if not pairs:
#             continue
#         current_parameters = {k: float(v) for k, v in pairs}

#         # Line contains parameters and is the start of a new block.
#         # Check if block matches parameter filter
#         if not parameter_filter:
#             # always match if filter is empty
#             parameters_match = True
#         else:
#             parameters_match = _parameters_are_close(
#                 parameter_filter, current_parameters,
#                 ensure_same_key_set=False
#             )

#         if parameters_match:
#             if any_previous_block_matches:
#                 raise CstAsciiParseError(f'Parameter filter is ambiguous, multiple matching blocks found in `{filename}`')

#             # parameters matched for the first time, start collecting lines
#             block_matches = True
#             any_previous_block_matches = True
#             block.clear()
#             block.append(line)
#             continue
#         else:
#             if block_matches:
#                 block.pop() # remove last line that was collected from next block
#             block_matches = False
        
#     # exiting loop, there should be one parsable block now
#     if not block:
#         raise CstAsciiParseError(f'Could not parse parameter combination `{parameter_filter}` from `{filename}`')

#     quantity = _get_quantity_from_cst_ascii_lines(
#         lines=block,
#         is_complex=is_complex,
#         convert_x_unit=convert_x_unit,
#         header_line_index=header_line_index
#     )
#     if not silent:
#         print(f'Parsed quantity, block has {len(block)} lines including headers.')
#         print(f'  First line: `{block[0][:30]} ...`')
#         print(f'  Last line:  `{block[-1][:30]} ...`')
    
#     return quantity


def get_quantity_from_cst_ascii(
    filename: Path | str,
    parameter_filter: dict[str, float] = {},
    is_complex: bool = True,
    convert_x_unit: bool = True,
    header_line_index: int = 1,
    silent: bool = True,
) -> RealQuantity | ComplexQuantity:
    """Select one trace matching parameters from a CST parametric sweep file. 
    The ASCII file is obtained by selecting an *parametric* impedance trace in CST and
    using Post-Processing > Import/Export > ASCII.

    Args:
        filename: Path to CST sweep ASCII file.
        parameter_filter: Matching criteria for parameters. Does not need to include all parameters
            from the CST project, just enough to unambigously identify the trace.
        is_complex: Wether the CST quantitiy is real or complex valued.
        convert_x_unit: If True, will infer X-axis unit from file and convert to base unit.
        header_line_index: Which line (starting with 0) in a block is the table header.
        silent: If False, print search status.

    Returns:
        Quantity matching the parameter filter. RealQuantity if `is_complex` is false,
        or ComplexQuantity `is_complex` is true.

    Raises:
        CstAsciiParseError: If parsing fails, if there is no match for `parameter_filter`,
            or if there is *more than one* match for `parameter_filter.`
    """
    # match lines in CST export against this to find parameters
    parameters_pattern = re.compile(r'(\w+)\s*=\s*([-+]?\d+(?:\.\d+)?)')
    no_parameters_pattern = re.compile(r'#$')

    block_slice_start: int | None = None
    block_slice_end: int | None = None
    data_after_block_start = False

    # first, iterate over entire file without actually parsing,
    # this will also ensure that parameter filter is unambiguous
    for n, line in enumerate(open(filename, 'r')):
        # No processing of data lines
        if not line.startswith('#'):
            if block_slice_start is not None:
                data_after_block_start = True
            continue

        # Line is a comment
        if (block_slice_start is not None) and data_after_block_start:
            block_slice_end = n

        if parameter_filter: # user has defined parameters to filter by
    
            pairs = parameters_pattern.findall(line)
            if not pairs:
                continue

            if not _parameters_are_close(
                parameter_filter,
                {k: float(v) for k, v in pairs},
                ensure_same_key_set=False
            ):
                continue

        else: # user has not provided parameter filter
            if not no_parameters_pattern.match(line):
                continue

        if block_slice_start is not None:
            raise CstAsciiParseError(f'Parameter filter `{parameter_filter}` is ambiguous, multiple matching blocks found in `{filename}`')
        
        block_slice_start = n
        data_after_block_start = False

    if block_slice_start is None:
        raise CstAsciiParseError(f'Could not parse parameter combination `{parameter_filter}` from `{filename}`')

    # actually load content into memory
    with open(filename, 'r') as fp:
        # for itertools.islice, None means start=0 or end=-1
        block = list(islice(fp, block_slice_start, block_slice_end))

    if not silent:
        print(f'Found quantity, block has {len(block)} lines including headers.')
        print(f'  First line: `{block[0][:30]} ...`')
        print(f'  Last line:  `{block[-1][:30]} ...`')
    
    quantity = _get_quantity_from_cst_ascii_lines(
        lines=block,
        is_complex=is_complex,
        convert_x_unit=convert_x_unit,
        header_line_index=header_line_index
    )
    
    return quantity