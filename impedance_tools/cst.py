from pathlib import Path
import re
import numpy as np

from .units import FREQUENCY_UNITS
from .quantities import Spectrum


class CstAsciiParseError(Exception):
    """Exception raised for errors parsing CST ASCII impedance data."""


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


def _get_impedance_from_cst_ascii_lines(lines: list[str]) -> Spectrum:
    """Parse impedance data from a list of CST ASCII export lines.
    Currently only Re/Im format is supported.

    Args:
        lines: Lines of text from a CST impedance export ASCII file.

    Returns:
        Spectrum object with frequency and complex impedance.

    Raises:
        CstAsciiParseError: When frequency unit or data format is invalid.
    """

    # get frequency unit and check export format
    freq_match = None
    header_line: str = ''
    for line in lines:
        freq_match = re.search('(' + '|'.join(FREQUENCY_UNITS.keys()) + ')', line)
        if freq_match is not None:
            header_line = line
            break

    # no line with frequency unit found
    if freq_match is None:
        raise CstAsciiParseError(f'Could not determine frequency unit')
    
    # parse frequency unit into factor
    try:    
        frequency_factor = FREQUENCY_UNITS[freq_match.group(1)]
    except KeyError:
        raise CstAsciiParseError(f'Unknown frequency unit: `{freq_match.group(1)}`')
    
    raw = np.loadtxt(lines, comments='#')

    # infer format of impedance from header line
    if re.search(r'(Re \/ Ohm).*(Im \/ Ohm)', header_line):
        return Spectrum(
            raw[:,0] * frequency_factor, # Hz
            raw[:,1] + 1j * raw[:,2] # Ohm
        )
    else:
        raise CstAsciiParseError(f'Could not determine format, or unknown format in header line `{header_line}`')
    

def get_impedance_from_cst_ascii(filename: Path | str) -> Spectrum:
    """Load a single impedance trace from a CST ASCII file.
    The ASCII file is obtained by selecting an **single** impedance trace in CST and
    using Post-Processing > Import/Export > ASCII.
    Currently only Re/Im format is supported.

    Args:
        filename: Path to a CST ASCII export impedance file.

    Returns:
        Spectrum object.
    """

    with open(filename) as fp:
        lines = fp.readlines()

    return _get_impedance_from_cst_ascii_lines(lines)


def get_all_impedances_from_cst_sweep_ascii(
    filename: Path | str,
    silent: bool = True,
) -> list[tuple[dict[str, float], Spectrum]]:
    """Load all impedances from a CST parametric sweep export ASCII file.
    The ASCII is obtained by selecting an **parametric** impedance trace in CST and
    using Post-Processing > Import/Export > ASCII.
    Currently only Re/Im format is supported.
    
    Args:
        filename: Path to CST parametric sweep ASCII file.
        silent: If False, print parsing status.

    Returns:
        List of (parameter dictionary, Spectrum object) tuples.

    Raises:
        CstAsciiParseError: If parameter block parsing fails.
    """

    parameter_pattern = re.compile(r'#Parameters\s*=\s*\{(.+?)\}')
    
    impedances: list[tuple[dict[str, float], Spectrum]] = []

    with open(filename) as fp:
        current_parameters: dict[str, float] | None = None
        current_lines: list[str] = []

        for line in fp:
            line = line.strip()
                
            # Check if this comment line has parameters
            re_match = parameter_pattern.search(line)
            if re_match:

                # process the previous block if it exists
                if current_parameters is not None:
                    impedances.append((
                        current_parameters,
                        _get_impedance_from_cst_ascii_lines(current_lines)
                    ))
                    if not silent:
                        print(f'Parsed impedance: {current_parameters}, block has {len(current_lines)} lines')
                    current_lines: list[str] = []

                # Parse parameters string into dictionary
                current_parameters = {}
                for param_str in re.split(r'[;,]', re_match.group(1)):
                    key_value = param_str.split('=')
                    if not len(key_value) == 2:
                        raise CstAsciiParseError(f'Error parsing parameter "{key_value}"')
                    
                    key = key_value[0].strip()
                    value = float(key_value[1].strip())

                    current_parameters[key] = value

            elif line and current_parameters is not None:
                # Here line is not with parameters, add to current block
                current_lines.append(line)

        # Add the last collected block if any
        if current_parameters is not None:
            impedances.append((
                current_parameters,
                _get_impedance_from_cst_ascii_lines(current_lines)
            ))
            if not silent:
                print(f'Parsed impedance: {current_parameters}, block has {len(current_lines)} lines')

        return impedances


def get_impedance_from_cst_sweep_ascii(
    filename: Path | str,
    parameter_filter: dict[str, float],
    silent: bool = True,
) -> Spectrum:
    """Select an impedance trace matching parameters from a CST parametric sweep file. 
    The ASCII file is obtained by selecting an **parametric** impedance trace in CST and
    using Post-Processing > Import/Export > ASCII.
    Currently only Re/Im format is supported.

    Args:
        filename: Path to CST sweep ASCII file.
        parameter_filter: Matching criteria for parameters. Does not need to include all parameters
            from the CST project, just enough to unambigously identify the trace.
        silent: If False, print search status.

    Returns:
        Matching Spectrum.

    Raises:
        CstAsciiParseError: If parsing fails, if there is no match for `parameter_filter`,
            or if there is *more than one* match for `parameter_filter.`
    """

    all_impedances: list[tuple[dict[str, float], Spectrum]] \
        = get_all_impedances_from_cst_sweep_ascii(filename, silent)

    # collect already scanned parameter combinations to check for duplicates
    scanned_parameters: list[dict[str, float]] = []
            
    # iterate over all impedances to find the matching one
    for parameters, impedance in all_impedances:

        if any(_parameters_are_close(parameters, scanned) for scanned in scanned_parameters):
            raise CstAsciiParseError(f'Duplicate parameter combination found')

        if all(
            key in parameters and _is_close(parameters[key], value)
            for key, value in parameter_filter.items()
        ):
            if not silent:
                print(f'Found matching impedance for parameters: {parameter_filter}')
            return impedance

    raise CstAsciiParseError(f'No impedance found matching parameters: {parameter_filter}')