# Short, consistent variable names

**Intent.** The package notation is verbose (`time_array`, `impedance_array`, `shunt_impedance`) and inconsistent between files (`beam_freq` next to `frequency_array`, `delta_t` next to `time_step`, `value_array` where an impedance is meant). Rename to short, canonical physics symbols, consistently across all modules, in signatures, properties, locals and tests.

**Goals.** One name per quantity everywhere. Zotter resonator notation. Tests follow.

**Non-goals.** Function and method names are unchanged. `cst.py`, `units.py`, `utils.py` are unchanged (already short). No behaviour change: every test keeps passing with the same assertions.

**Anti-goals.** Partial renames that leave two names for one quantity. Changing numeric results.

## Rules

- Axes and signals get single letters: `t`, `f`, `w`, `z`, `dt`, `df`, `omega`.
- Qualifiers go as suffix on short symbols: `t_bunch`, `f_beam`, `f_max`, `f_cutoff`, `w_convolved`.
- Resonator symbols follow Zotter: `f_r`, `omega_r`, `R`, `Q`, `R_over_Q`, `alpha`, `omega_reduced`, `Q_reduced`. Capitals `R`, `Q` are kept deliberately as canonical physics symbols (precedent: `scipy.signal.StateSpace.A/B/C/D`). If pep8-naming rules are ever enabled in ruff, ignore N803, N806, N815.
- Inside a class the class noun is dropped: `Bunch.full_length`, not `full_bunch_length`.
- `h` is the RF harmonic number (`Beam.harmonic_number` stays as is), `n` is the revolution-harmonic index array.
- Already-short words without a standard symbol stay: `sigma`, `charge`, `profile`, `spectrum`, `half_window`, `wake_length`, `zero_time_tolerance`.

## Approaches considered

1. Rename names only, keep `_Resonator` with private attributes and passthrough properties. Minimal diff, but `_f_r`/`f_r` pairs are pure noise once names are short.
2. **Chosen.** Rename and make `_Resonator` a frozen dataclass with fields `f_r`, `R`, `Q` and a `__post_init__` check on `Q`, matching how `Bunch` is built. Derived quantities stay properties.

## Rename table

| Current | New | Where |
|---|---|---|
| `time_array` | `t` | all |
| `frequency_array` | `f` | all |
| `wake_array`, `wake_function`, `wake_potential` | `w` | all |
| `impedance_array`, `impedance`, `value_array` | `z` | all |
| `time_step`, `delta_t` | `dt` | beam, incl. `_check_sampling(dt=...)`, `_get_internal_time_array(dt)` |
| `frequency_step` | `df` | beam, incl. `Bunch.get_numeric_spectrum(f_max, df)` |
| `max_frequency` | `f_max` | `Beam` field, `Bunch.get_numeric_spectrum` |
| `beam_freq` | `f_beam` | beam |
| `real_impedance_at_beam_freq` | `real_z_at_f_beam` | beam |
| `bunch_time_array` | `t_bunch` | beam, wake |
| `beam_time_array`, `beam_profile` | `t`, `profile` | `Beam.get_profile`, `Beam.get_numeric_spectrum` |
| `spectrum_array` | `spectrum` | beam |
| `harmonic_array`, `max_harmonic` | `n`, `n_max` | beam |
| `nyquist_frequency`, `cutoff_frequency` | `f_nyquist`, `f_cutoff` | beam, wake |
| `max_impedance_delta_f` | `max_df` | beam |
| `max_frequency_shift` | `max_shift` | `get_shifted_power_loss` |
| `power_losses` (local) | `power_loss` | beam (`power_loss_spectrum` is kept, it matches the method name) |
| `negative_real_impedance_handling` | `on_negative_impedance` | beam |
| `loss_factor` (local) | `k` | beam |
| `resonant_frequency` | `f_r` | ctor arg and field |
| `shunt_impedance` | `R` | ctor arg and field |
| `quality_factor` | `Q` | ctor arg and field |
| `damping_coefficient` | `alpha` | property |
| `reduced_omega_r` | `omega_reduced` | property |
| `reduced_quality_factor` | `Q_reduced` | property |
| `bunch_length_sigma`, `bunch_sigma` | `sigma` | `LongitudinalResonator.get_loss_factor`, `get_impedance_from_wake` |
| `cutoff_by_bunch_sigma` | `cutoff_sigma` | `get_impedance_from_wake` |
| `exponent` | `mu` | `BinomialBunch` field |
| `full_bunch_length` | `full_length` | `BinomialBunch` |
| `oversampling_factor` | `oversampling` | wake |
| `start_time` | `t_start` | wake |
| `time_period` | `period` | `Bunch.get_numeric_spectrum` |

Left to the builder: names of purely local helpers not listed here, chosen by the same rules.
