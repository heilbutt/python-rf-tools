from contextlib import contextmanager
from dataclasses import dataclass

__all__ = [
    'settings',
    'settings_context'
]


@dataclass(slots=True) # slots prevent accidental creating of new attributes
class _Settings:
    sanity_check: bool = False
    sanity_check_tolerance: float = 1e-5

settings = _Settings()

@contextmanager
def settings_context(**kwargs):
    old_config = {k: getattr(settings, k) for k in kwargs}
    for key, value in kwargs.items():
        setattr(settings, key, value)
    try:
        yield
    finally:
        for key, value in old_config.items():
            setattr(settings, key, value)