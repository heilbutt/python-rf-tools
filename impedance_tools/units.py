from scipy.constants import foot, mile ,inch

FREQUENCY_UNITS: dict[str, float] = {
    'Hz' : 1e0,
    'kHz': 1e3,
    'MHz': 1e6,
    'GHz': 1e9,
}

LENGTH_UNITS: dict[str, float] = {
    'm'  : 1e0,
    'cm' : 1e-2,
    'mm' : 1e-3,
    'um' : 1e-6,
    'nm' : 1e-9,
    'ft' : foot,
    'mil': mile,
    'in' : inch
}