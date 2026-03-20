from pathlib import Path
import re
import numpy as np

from .units import FREQUENCY_UNITS
from .quantities import Impedance


def _is_close(a: float, b: float, rel_tol: float = 1e-09, abs_tol: float = 0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def _parameters_are_close(
    a: dict[str, float], b: dict[str, float],
    rel_tol: float = 1e-09, abs_tol: float = 0.0
) -> bool:
    
    for k in a.keys():
        if not _is_close(a[k], b[k], rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


def _get_impedance_from_cst_ascii_lines(lines: list[str]) -> Impedance:

    
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
        raise RuntimeError(f'Could not determine frequency unit')
    
    # parse frequency unit into factor
    try:    
        frequency_factor = FREQUENCY_UNITS[freq_match.group(1)]
    except KeyError:
        raise RuntimeError(f'Unknown frequency unit: `{freq_match.group(1)}`')
    
    raw = np.loadtxt(lines, comments='#')

    # infer format of impedance from header line
    if re.search(r'(Re \/ Ohm).*(Im \/ Ohm)', header_line):
        return Impedance(
            raw[:,0] * frequency_factor, # Hz
            raw[:,1] + 1j * raw[:,2] # Ohm
        )
    else:
        raise RuntimeError(f'Could not determine format, or unknown format in header line `{header_line}`')
    

def get_impedance_from_cst_ascii(filename: Path | str) -> Impedance:
    
    if isinstance(filename, (Path, str)):
        with open(filename) as fp:
            lines = fp.readlines()

    return _get_impedance_from_cst_ascii_lines(lines)


def get_all_impedances_from_cst_sweep_ascii(
    filename: Path | str,
    silent: bool = True,
) -> list[tuple[dict[str, float], Impedance]]:
    
    parameter_pattern = re.compile(r'#Parameters\s*=\s*\{(.+?)\}')
    
    impedances: list[tuple[dict[str, float], Impedance]] = []

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
                        raise RuntimeError(f'Error parsing parameter "{key_value}"')
                    
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
) -> Impedance:
    
    all_impedances: list[tuple[dict[str, float], Impedance]] \
        = get_all_impedances_from_cst_sweep_ascii(filename, silent)

    # collect already scanned parameter combinations to check for duplicates
    scanned_parameters: list[dict[str, float]] = []
            
    # iterate over all impedances to find the matching one
    for parameters, impedance in all_impedances:

        if any(_parameters_are_close(parameters, scanned) for scanned in scanned_parameters):
            raise RuntimeError(f'Duplicate parameter combination found')

        if all(
            key in parameters and _is_close(parameters[key], value)
            for key, value in parameter_filter.items()
        ):
            if not silent:
                print(f'Found matching impedance for parameters: {parameter_filter}')
            return impedance

    raise RuntimeError(f'No impedance found matching parameters: {parameter_filter}')