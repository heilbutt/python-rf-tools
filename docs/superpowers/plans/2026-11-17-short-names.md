# Short Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename parameters, properties, fields and locals across `rf_tools` to short canonical physics symbols, consistently, with tests updated to match.

**Architecture:** Pure rename, no behaviour change. `_Resonator` becomes a frozen dataclass with fields `f_r`, `R`, `Q` (like `Bunch`). Each task renames one module region and its tests, verified by the unchanged test assertions.

**Tech Stack:** Python 3.12, numpy, scipy, pytest, ruff, pyrefly, run via `uv run`.

**Spec:** `docs/superpowers/specs/2026-11-17-short-names-design.md` (contains the full rename table; the tasks below reference it).

## Global Constraints

- Function and method names do not change. Only parameters, properties, dataclass fields and locals.
- Unit comments (`# s`, `# Hz`, `# Ohm/m^n`) stay next to the renamed names.
- `cst.py`, `units.py`, `utils.py`, `__init__.py`, `test_cst.py`, `test_units.py`, `test_utils.py`, `test_exports.py` are not touched.
- The working tree already contains an uncommitted partial rename of `src/rf_tools/wake.py` and `tests/test_wake.py`. Build on it, do not revert it.
- Baseline before work: `uv run pytest -q` gives 218 passed; `uv run pyrefly check` gives 0 errors; `uv run ruff check` gives 19 pre-existing errors (16 × C408, 3 × I001). Do not add new ruff errors; do not fix the pre-existing ones (out of scope).
- Test assertions and numeric tolerances do not change. Only names in tests change.
- Inside tests, locals follow the same rules: `wake` → `w`, `impedance` → `z`, `wake_potential` → `w_potential`, `wake_back` → `w_back`, `w1`/`w2`/`w_conv` stay. Test parametrize names change with the parameter they feed (`quality_factor` → `Q`, `exponent` → `mu`, `start_time` → `t_start`, `max_frequency` → `f_max`). Test constants like `F_R`, `R_SHUNT`, `MAX_FREQUENCY` stay.
- For every task, after the test file is changed and before the source is changed, the changed tests must fail with `TypeError` (unexpected keyword) or `AttributeError`, never with an assertion error.

## Review Focus

- `BinomialBunch(charge, sigma, mu)` positional order unchanged: `BinomialBunch(1, SIGMA, 1.5)` must still build. Pinned by existing `BUNCHES` params in `tests/test_beam.py`.
- `LongitudinalResonator(F_R, R_SHUNT, 10)` positional order `f_r, R, Q` unchanged. Pinned by `test_resonator_properties`.
- The dataclass `_Resonator` must still reject `Q <= 0.5` with `ValueError`. Pinned by `test_resonator_rejects_overdamped`.
- `Beam(..., f_max=...)` keyword and `replace(beam, f_max=1e9)` must work; the frozen dataclass must still reject attribute assignment. Pinned by `test_beam_is_immutable`, `test_beam_equality_and_replace`.
- `LongitudinalResonator` and `TransverseResonator` with equal fields must not compare equal to each other (dataclass `__eq__` compares class). Add the test in Task 1.

---

### Task 1: Resonator classes in `wake.py`

**Files:**
- Modify: `src/rf_tools/wake.py` (class `_Resonator`, `LongitudinalResonator`, `TransverseResonator`, lines ~127–286)
- Test: `tests/test_wake.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class _Resonator: f_r: float; R: float; Q: float` with `__post_init__` raising `ValueError` if not `Q > 0.5`.
  - Properties: `omega_r`, `R_over_Q`, `alpha`, `Q_reduced`, `omega_reduced` (same formulas as today's `omega_r`, `R_over_Q`, `damping_coefficient`, `reduced_quality_factor`, `reduced_omega_r`).
  - `LongitudinalResonator.get_loss_factor(self, sigma: float = 0) -> float`
  - `LongitudinalResonator.get_impedance(self, f, wake_length=None)` and `TransverseResonator.get_impedance(self, f, wake_length=None)`
  - `LongitudinalResonator.get_wake_function(self, t, zero_time_tolerance=1e-15)` and `TransverseResonator.get_wake_function(self, t, zero_time_tolerance=1e-15)`
  - Locals inside these methods: `z` (was `impedance`), `w` (was `wake_function`).

- [ ] **Step 1: Update the resonator tests in `tests/test_wake.py`**

Rename in the test file: `r.resonant_frequency` → `r.f_r`, `r.shunt_impedance` → `r.R`, `r.quality_factor` → `r.Q`, `r.damping_coefficient` → `r.alpha`, `r.reduced_quality_factor` → `r.Q_reduced`, `r.reduced_omega_r` → `r.omega_reduced`, the keyword `bunch_length_sigma=` → `sigma=`, parametrize name `quality_factor` → `Q` (in all four parametrized tests and `wake_time_axis`). Rename test locals `impedance` → `z` and `wake` → `w` in the resonator tests. Add:

```python
def test_resonators_of_different_type_are_not_equal():
    assert LongitudinalResonator(F_R, R_SHUNT, 10) == LongitudinalResonator(F_R, R_SHUNT, 10)
    assert LongitudinalResonator(F_R, R_SHUNT, 10) != TransverseResonator(F_R, R_SHUNT, 10)
```

- [ ] **Step 2: Run the tests to verify they fail on names**

Run: `uv run pytest tests/test_wake.py -q -x`
Expected: FAIL with `AttributeError` (e.g. `'LongitudinalResonator' object has no attribute 'f_r'`).

- [ ] **Step 3: Rewrite the resonator classes in `src/rf_tools/wake.py`**

Replace `_Resonator.__init__` and the three storage properties with a frozen dataclass (`from dataclasses import dataclass`) holding `f_r`, `R`, `Q` with unit comments, keep the `Q > 0.5` check in `__post_init__` with the existing message text (substituting `Q`). Rename the derived properties and all their uses in both subclasses per the Interfaces block. Rename method parameters `frequency_array` → `f`, `time_array` → `t`, `bunch_length_sigma` → `sigma`, and locals `impedance` → `z`, `wake_function` → `w`. The warning message in `get_loss_factor` keeps its wording but reads `omega_r * sigma`.

- [ ] **Step 4: Run tests, lint, types**

Run: `uv run pytest -q && uv run pyrefly check && uv run ruff check | tail -1`
Expected: 219 passed; pyrefly 0 errors; ruff still `Found 19 errors.`

- [ ] **Step 5: Commit**

```bash
git add src/rf_tools/wake.py tests/test_wake.py
git commit -m "Short resonator names: f_r, R, Q, alpha, omega_reduced, Q_reduced"
```

---

### Task 2: Wake/impedance functions in `wake.py`

**Files:**
- Modify: `src/rf_tools/wake.py` (`get_impedance_from_wake`, `get_wake_from_impedance`, `convolve_wake`, lines ~26–124)
- Test: `tests/test_wake.py`

**Interfaces:**
- Consumes: `GaussianBunch._check_sampling(time_step=...)` and `_get_internal_time_array(time_step=...)` still take `time_step` until Task 3; keep those keyword calls for now (Task 3 updates them).
- Produces:
  - `get_impedance_from_wake(t, w, oversampling: float = 1, sigma: float | None = None, cutoff_sigma: float | None = None) -> tuple[Real1dArray, Complex1dArray]`
  - `get_wake_from_impedance(f, z, oversampling: float = 1, t_start: float = 0) -> tuple[Real1dArray, Real1dArray]`
  - `convolve_wake(t, w, target_sigma, wake_sigma=0)` (already done in the working tree; locals `bunch_time_array` → `t_bunch`, `bunch_profile` stays).
  - Locals in `get_impedance_from_wake`: `frequency_array` → `f`, `impedance_array` → `z`, `cutoff_frequency` → `f_cutoff`.

- [ ] **Step 1: Update the function tests in `tests/test_wake.py`**

Keywords: `oversampling_factor=` → `oversampling=`, `bunch_sigma=` → `sigma=`, `cutoff_by_bunch_sigma=` → `cutoff_sigma=`, `start_time=` → `t_start=` (also the parametrize name and the local in `test_wake_impedance_roundtrip`). Rename test locals `wake` → `w`, `impedance` → `z`, `wake_potential` → `w_potential`, `wake_back` → `w_back` in the remaining tests; the helper `step_wake()` returns `(t, w)`. Error-message `match=` strings, if any reference `bunch_sigma`, follow the new wording in Step 3.

- [ ] **Step 2: Run the tests to verify they fail on names**

Run: `uv run pytest tests/test_wake.py -q -x`
Expected: FAIL with `TypeError: ... unexpected keyword argument`.

- [ ] **Step 3: Rename inside the three functions in `src/rf_tools/wake.py`**

Apply the Interfaces block. The `ValueError` in `get_impedance_from_wake` becomes `'Cannot cutoff by bunch sigma if `sigma` is not provided'`.

- [ ] **Step 4: Run tests, lint, types**

Run: `uv run pytest -q && uv run pyrefly check && uv run ruff check | tail -1`
Expected: 219 passed; pyrefly 0 errors; ruff `Found 19 errors.`

- [ ] **Step 5: Commit**

```bash
git add src/rf_tools/wake.py tests/test_wake.py
git commit -m "Short names in wake/impedance functions: t, w, f, z, sigma, cutoff_sigma, t_start"
```

---

### Task 3: `Bunch` classes in `beam.py`

**Files:**
- Modify: `src/rf_tools/beam.py` (`Bunch`, `GaussianBunch`, `BinomialBunch`, lines ~40–262)
- Modify: `src/rf_tools/wake.py` (`convolve_wake` calls `_check_sampling` / `_get_internal_time_array`)
- Test: `tests/test_beam.py`

**Interfaces:**
- Produces:
  - `Bunch._get_normalized_profile(self, t)`, `get_form_factor(self, f)`, `get_profile(self, t)`
  - `Bunch._get_internal_time_array(self, dt: float)`
  - `Bunch._check_sampling(self, dt: float, half_window: float | None = None, stacklevel: int = 3)`
  - `Bunch.get_numeric_spectrum(self, f_max: float, df: float) -> tuple[Real1dArray, Complex1dArray]` with locals `period` (was `time_period`), `dt`, `t`, `spectrum`, `f`.
  - `Bunch.get_loss_factor(self, t, w) -> float`: the zero-padded arrays reassign `t` and `w` (as today's code reassigns `time_array`), result local is `k`.
  - `_check_sampling` local `nyquist_frequency` → `f_nyquist`; warning text `Time step {format_quantity(dt, "s")} ...` unchanged otherwise.
  - `BinomialBunch` field `mu: float` (was `exponent`), property `full_length` (was `full_bunch_length`). Error message: `f'Binomial exponent mu must be non-negative, got {self.mu}'`.
  - `BinomialBunch._get_form_factor_envelope` local `frequency_array` → `f`.

- [ ] **Step 1: Update bunch tests in `tests/test_beam.py`**

`max_frequency=` → `f_max=` and `frequency_step=` → `df=` in all `get_numeric_spectrum` calls; `full_bunch_length` → `full_length`; parametrize `exponent` → `mu` in `test_binomial_profile_has_finite_support` (keep the id strings). Rename local `wake` → `w` in `test_bunch_loss_factor_pads_short_wake_with_zeros`. Leave `Beam` tests for Task 4.

- [ ] **Step 2: Run to verify failure on names**

Run: `uv run pytest tests/test_beam.py -q -x -k "bunch or binomial or gaussian"`
Expected: FAIL with `TypeError` or `AttributeError`.

- [ ] **Step 3: Rename inside the `Bunch` classes in `src/rf_tools/beam.py` and update the two calls in `convolve_wake`**

Apply the Interfaces block. In `convolve_wake`: `bunch._check_sampling(dt=dt)`, `t_bunch = bunch._get_internal_time_array(dt=dt)`.

- [ ] **Step 4: Run tests, lint, types**

Run: `uv run pytest -q && uv run pyrefly check && uv run ruff check | tail -1`
Expected: 219 passed; pyrefly 0 errors; ruff `Found 19 errors.`

- [ ] **Step 5: Commit**

```bash
git add src/rf_tools/beam.py src/rf_tools/wake.py tests/test_beam.py
git commit -m "Short names in Bunch classes: t, f, w, dt, df, f_max, mu, full_length"
```

---

### Task 4: `Beam` class in `beam.py`

**Files:**
- Modify: `src/rf_tools/beam.py` (`Beam`, lines ~265–576)
- Test: `tests/test_beam.py`

**Interfaces:**
- Produces:
  - `Beam` field `f_max: float` (was `max_frequency`), same position (second field).
  - `Beam.get_profile()` locals: `dt`, `t_bunch`, `bunch_profile`, `bucket_start_times`, returns `(t, profile)`.
  - `Beam._get_harmonic_numbers()` local `n_max`.
  - `get_spectrum`, `_get_squared_spectrum`: locals `n` (was `harmonic_array`), `f`, `spectrum`; `filling_pattern_spectrum`, `filling_pattern_power`, `squared_spectrum` stay.
  - `get_numeric_spectrum`: locals `t, profile = self.get_profile()`, `dt`, `spectrum`, `f`.
  - `_get_real_impedance_at_beam_frequencies(self, f_beam, f, z, on_negative_impedance) -> Real1dArray`, locals `max_df`, `f_upper` (was the local `max_frequency`; named `f_upper` so it is not confused with the field `self.f_max`), `real_z_at_f_beam`.
  - `get_power_loss_spectrum(self, f, z, on_negative_impedance: NegativeImpedanceHandling = 'raise')` returns `(f_beam, power_loss_spectrum)`.
  - `get_power_loss(self, f, z, on_negative_impedance='raise') -> float`.
  - `get_shifted_power_loss(self, f, z, max_shift: float, on_negative_impedance='raise')`, locals `df` (was `frequency_step`), `power_loss` (was `power_losses`).
  - Error message for unknown handling: ``f'Unknown handling of negative impedance real part: `{on_negative_impedance}`'``.

- [ ] **Step 1: Update beam tests in `tests/test_beam.py`**

`max_frequency=` → `f_max=` in `make_beam`, `replace`, parametrize name and the immutability test (`beam.f_max = 1e9`); `negative_real_impedance_handling=` → `on_negative_impedance=`; `max_frequency_shift=` → `max_shift=`; rename test local `impedance` → `z` in the power-loss tests and in `broadband_impedance`.

- [ ] **Step 2: Run to verify failure on names**

Run: `uv run pytest tests/test_beam.py -q -x`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'f_max'`.

- [ ] **Step 3: Rename inside `Beam` in `src/rf_tools/beam.py`**

Apply the Interfaces block.

- [ ] **Step 4: Run tests, lint, types, and grep for leftovers**

Run: `uv run pytest -q && uv run pyrefly check && uv run ruff check | tail -1`
Expected: 219 passed; pyrefly 0 errors; ruff `Found 19 errors.`

Run: `grep -rnE "time_array|frequency_array|impedance_array|wake_array|time_step|frequency_step|max_frequency|beam_freq|shunt_impedance|quality_factor|resonant_frequency|damping_coefficient|reduced_|bunch_sigma|cutoff_by|exponent|full_bunch_length|oversampling_factor|start_time|negative_real|harmonic_array|value_array|delta_t|time_period" src tests`
Expected: only hits in `src/rf_tools/cst.py`/`tests/test_cst.py` (none expected there either), and `_get_internal_time_array` (method name, kept). Anything else is a leftover to fix.

- [ ] **Step 5: Commit**

```bash
git add src/rf_tools/beam.py tests/test_beam.py
git commit -m "Short names in Beam: f_max, f, z, n, on_negative_impedance, max_shift"
```
