from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from itertools import islice

from .units import PREFIXES

from typing import Literal, Iterable, overload
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


@dataclass
class _BlockMetadata:
    start_index: int # inclusive, 0-based
    stop_index: int | None # exclusive, 0-based, None means EOF
    header_lines: list[str]


def _get_block_metadata_from_cst_ascii(
    filename: Path | str,
) -> list[_BlockMetadata]:
    
    blocks: list[_BlockMetadata] = []
    current_header_lines: list[str] = []
    current_block_start: int | None = None
    have_seen_data_in_current_block: bool = False

    for line_number, line in enumerate(open(filename, 'r')):
        if not line.startswith('#'):
            # data line
            have_seen_data_in_current_block = True
            continue # not a comment line

        # header line
        if not have_seen_data_in_current_block:
            current_header_lines.append(line)
            if current_block_start is None:
                current_block_start = line_number
            continue

        # start of a new block, close the current block
        if current_block_start is None:
            raise CstAsciiParseError(f'Could not parse blocks in `{filename}`')
        blocks.append(_BlockMetadata(
            start_index=current_block_start,
            stop_index=line_number,
            header_lines=current_header_lines.copy()
        ))
        current_block_start = line_number
        current_header_lines = [line]
        have_seen_data_in_current_block = False

    # close last block if file ended without a new header
    if current_block_start is None:
        raise CstAsciiParseError(f'Could not parse blocks in `{filename}`')
    blocks.append(_BlockMetadata(
        start_index=current_block_start,
        stop_index=None,
        header_lines=current_header_lines.copy()
    ))

    return blocks


def _get_complex_quantity_from_cst_ascii_lines(
    lines: Iterable[str],
) -> tuple[RealArray, ComplexArray]:
    
    raw = np.loadtxt(lines, comments='#')

    for line in lines:
        if not line.startswith('#'):
            continue

        # Try format Real-imaginary
        if re.search(r'(\[Re).*(\[Im)', line):
            return (
                raw[:,0],
                raw[:,1] + 1j * raw[:,2]
            )
    
        # Try format Magnitude-phase
        if re.search(r'(\[Mag).*(\[Pha)', line):
            return (
                raw[:,0],
                np.asarray(raw[:,1], dtype=complex) * np.exp(1j * raw[:,2])
            )

    raise CstAsciiParseError(f'Could not determine complex format. Is this a complex-valued export from CST?')
    

def _get_real_quantity_from_cst_ascii_lines(
    lines: Iterable[str],
) -> tuple[RealArray, RealArray]:

    raw = np.loadtxt(lines, comments='#')
    return (
        raw[:,0],
        raw[:,1]
    )
    

@overload
def get_quantity_from_cst_ascii(
    filename: Path | str,
    is_complex: Literal[True] = ...,
    parameter_filter: dict[str, float] = ...,
    x_multiplier: float | str = ...,
    y_multiplier: float | str = ...,
    silent: bool = ...,
) -> tuple[RealArray, ComplexArray]:
    ...
    
@overload
def get_quantity_from_cst_ascii(
    filename: Path | str,
    is_complex: Literal[False],
    parameter_filter: dict[str, float] = ...,
    x_multiplier: float | str = ...,
    y_multiplier: float | str = ...,
    silent: bool = ...,
) -> tuple[RealArray, RealArray]:
    ...

def get_quantity_from_cst_ascii(
    filename: Path | str,
    is_complex: bool = True,
    parameter_filter: dict[str, float] = {},
    x_multiplier: float | str = 1,
    y_multiplier: float | str = 1,
    silent: bool = True,
) -> tuple[RealArray, RealArray | ComplexArray]:

    # first, iterate over entire file without actually parsing,
    # this will also ensure that parameter filter is unambiguous

    blocks = _get_block_metadata_from_cst_ascii(filename)
    if not silent:
        print(f'Found {len(blocks)} data blocks, with the following line ranges (1-based, inclusive):')
        for n, block in enumerate(blocks):
            block_end_index = 'EOF' if block.stop_index is None else str(block.stop_index)
            print(f'  Block {n+1:3d}: lines {block.start_index+1:6d} to {block_end_index:>6}')

    if not parameter_filter:
        # if user specifies no parameter filter, there must be exactly one block
        if not len(blocks) == 1:
            raise CstAsciiParseError(f'No parameter filter specified, but {len(blocks)} blocks found in `{filename}`') 
        matching_block = blocks[0]
    else:
        # if user specified parameter filter, there must be exactly one matching block
        # regular expression to find parameter-value pairs
        pattern = re.compile(r'(\w+)\s*=\s*([-+]?\d+(?:\.\d+)?)')
        
        matching_blocks: list[_BlockMetadata] = []
        for block in blocks:
            for header_line in block.header_lines:
                try:
                    if _parameters_are_close(
                        parameter_filter,
                        {k: float(v) for k, v in pattern.findall(header_line)},
                        ensure_same_key_set=False
                    ):
                        matching_blocks.append(block)
                        break
                except KeyError:
                    continue

        if len(matching_blocks) > 1:
            raise CstAsciiParseError(f'Parameter filter `{parameter_filter}` is ambiguous, multiple matching blocks found in `{filename}`')
        if len(matching_blocks) < 1:
             raise CstAsciiParseError(f'Parameter filter `{parameter_filter}` delivered no matches in `{filename}`')
        matching_block = matching_blocks[0]

    # actually load content into memory and parse it
    with open(filename, 'r') as fp:
        # for itertools.islice, `None` means full range
        block_lines = list(islice(fp, matching_block.start_index, matching_block.stop_index))
    
    if not silent:
        print(f'Found quantity, block has {len(block_lines)} lines including headers')
        print(f'  First line: `{block_lines[0][:50]} ...`')
        print(f'  Last line:  `{block_lines[-1][:50]} ...`')
    
    if is_complex:
        x, y = _get_complex_quantity_from_cst_ascii_lines(block_lines)
    else:
        x, y = _get_real_quantity_from_cst_ascii_lines(block_lines)

    if isinstance(x_multiplier, str):
        try:
            x_multiplier = PREFIXES[x_multiplier]
        except KeyError:
            raise CstAsciiUnitParseError(f'Unknown x-axis multiplier `{x_multiplier}`')
    if isinstance(y_multiplier, str):
        try:
            y_multiplier = PREFIXES[y_multiplier]
        except KeyError:
            raise CstAsciiUnitParseError(f'Unknown y-axis multiplier `{y_multiplier}`')
        
    return (
        x * x_multiplier,
        y * y_multiplier
    )