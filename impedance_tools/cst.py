from pathlib import Path
import re
import numpy as np

from .units import TIME_UNITS, FREQUENCY_UNITS, LENGTH_UNITS
from .quantities import RealQuantity, ComplexQuantity

from typing import Sequence


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

    if is_complex:
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
    
    else:
        raw = np.loadtxt(lines, comments='#')
        return RealQuantity(
            raw[:,0] * x_unit_factor,
            raw[:,1]
        )
    

def get_quantity_from_cst_ascii(
    filename: Path | str,
    is_complex: bool = True,
    convert_x_unit: bool = True,
    header_line_index: int = 1,
) -> RealQuantity | ComplexQuantity:
    
    with open(filename) as fp:
        lines = fp.readlines()
    return _get_quantity_from_cst_ascii_lines(
        lines=lines,
        is_complex=is_complex,
        convert_x_unit=convert_x_unit,
        header_line_index=header_line_index
    )


def get_all_quantities_from_cst_sweep_ascii(
    filename: Path | str,
    is_complex: bool = True,
    convert_x_unit: bool = True,
    header_line_index: int = 1,
    silent: bool = True,
) -> list[tuple[dict[str, float], RealQuantity]] | list[tuple[dict[str, float], ComplexQuantity]]:
    """Load all quantities from a CST parametric sweep export ASCII file.
    The ASCII is obtained by selecting an *parametric* impedance trace in CST and
    using Post-Processing > Import/Export > ASCII.
    
    Args:
        filename: Path to CST parametric sweep ASCII file.
        is_complex: Wether the CST quantitiy is real or complex valued.
        convert_x_unit: If True, will infer X-axis unit from file and convert to base unit.
        header_line_index: Which line (starting with 0) in a block is the table header.
        silent: If False, print parsing status.

    Returns:
        List of (parameter dictionary, RealQuantity) tuples if `is_complex` is false,
        or list of (parameter dictionary, ComplexQuantity) tuples if `is_complex` is true

    Raises:
        CstAsciiParseError: If parameter block parsing fails.
    """

    parameter_pattern = re.compile(r'#Parameters\s*=\s*\{(.+?)\}')
    
    quantities = []

    with open(filename) as fp:

        current_parameters: dict[str, float] | None = None
        current_lines: list[str] = []
        blocks: list[tuple[dict[str, float], list[str]]] = []

        for line in fp:
            # Check if this line has parameters
            parameter_match = parameter_pattern.search(line)
            
            if parameter_match:
                # attach the the previous block if it exists
                if current_parameters is not None:
                    blocks.append((current_parameters, current_lines))
                    current_lines = []

                # Parse parameters string into dictionary
                current_parameters = {}
                for param_str in re.split(r'[;,]', parameter_match.group(1)):
                    key_value = param_str.split('=')
                    if not len(key_value) == 2:
                        raise CstAsciiParseError(f'Error parsing parameter "{key_value}"')
                    
                    key = key_value[0].strip()
                    value = float(key_value[1].strip())

                    current_parameters[key] = value

            if line:
                current_lines.append(line)

        # Add the last collected block at the end of the for-loop
        if current_parameters is not None:
            blocks.append((current_parameters, current_lines))

        # process collected blocks
        for block in blocks:
            quantities.append((
                block[0],
                _get_quantity_from_cst_ascii_lines(
                    lines=block[1],
                    is_complex=is_complex,
                    convert_x_unit=convert_x_unit,
                    header_line_index=header_line_index
                )
            ))
            if not silent:
                print(f'Parsed quantity: {current_parameters}, block has {len(current_lines)} lines')

        return quantities


def get_quantity_from_cst_sweep_ascii(
    filename: Path | str,
    parameter_filter: dict[str, float],
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

    all_quantities = get_all_quantities_from_cst_sweep_ascii(
        filename=filename,
        is_complex=is_complex,
        convert_x_unit=convert_x_unit,
        header_line_index=header_line_index,
        silent=silent
    )

    # collect already scanned parameter combinations to check for duplicates
    scanned_parameters: list[dict[str, float]] = []
            
    # iterate over all impedances to find the matching one
    for parameters, quantity in all_quantities:

        if any(_parameters_are_close(parameters, scanned) for scanned in scanned_parameters):
            raise CstAsciiParseError(f'Duplicate parameter combination found')

        if all(
            key in parameters and _is_close(parameters[key], value)
            for key, value in parameter_filter.items()
        ):
            if not silent:
                print(f'Found matching quantity for parameters: {parameter_filter}')
            return quantity

    raise CstAsciiParseError(f'No quantity found matching parameters: {parameter_filter}')