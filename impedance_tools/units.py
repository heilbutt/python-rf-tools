PREFIXES: dict[str, float] = {
    'f' : 1e-15,
    'p' : 1e-12,
    'n' : 1e-9,
    'u' : 1e-6,
    'm' : 1e-3,
    ''  : 1e+0,
    'k' : 1e+3,
    'M' : 1e+6,
    'G' : 1e+9,
    'T' : 1e+12
}

TIME_UNITS = {key + 's': value for key, value in PREFIXES.items()}
FREQUENCY_UNITS = {key + 'Hz': value for key, value in PREFIXES.items()}
LENGTH_UNITS = {key + 'm': value for key, value in PREFIXES.items()}