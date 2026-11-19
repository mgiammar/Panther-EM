# After-CTF noise normalization (`normalization: after_ctf`)

**Status:** the default, after the validation in section 5. `normalization: before_ctf`
restores Panther's original behaviour.

> **Behaviour change.** Searches now report different z-scores than before for the same
> micrograph: particle z is unchanged on average (+0.02 to +0.07) but each hypothesis's
> noise level is evened out, so the noise maximum is lower. Null calibrations, FDR cuts,
> refine-and-rescore cuts and hard-coded z thresholds made with earlier versions do not
> carry over. Re-run the nulls, or set `normalization: before_ctf` to reproduce old results
> exactly. Each result file records the setting (the search config is stored with it).

**In one sentence:** every `(defocus plane, template)` hypothesis is rescaled so that its
correlation has the same noise variance under its CTF, as Leopard-EM and cisTEM do when
they normalize each CTF-filtered projection; particle z-scores stay the same on average,
the noise-only maximum comes down, and the null-calibrated threshold drops with it.

---

## 1. The problem: Panther normalizes before the CTF

Panther correlates the image with every template projection `m_n` (row `n` = orientation
x Fourier filter) at every searched defocus plane `d`. It normalizes the whitened
projection to unit variance **before** the CTF, and applies the CTF to the image (one
filtered copy of the image's features per plane). The noise variance of hypothesis
`(d, n)` is then not constant:

```
e[d, n] = sum_F  noise[d, F] * P[n, F]
```

* `P[n, F]` – ring power of template `n` (its error table's `reconstruction_autocorr`);
* `noise[d, F]` – the micrograph's noise power under `CTF_d` (`Micrograph.noise`).

`e` is exactly the stage-1 threshold test's `q_kept` (at unit local noise level). It varies

* with `n`: how much of a projection's power falls where the CTF passes it, and
* with `d`: how much power the plane's CTF passes at all.

On xenon_131 (13 planes, -1200 to +1200 A, nside 32 x 256 psi, rank 96 x 80):

| template | sd of `e` over all (plane, template) | per-template part only | after this change |
|---|---|---|---|
| 60S, B 0 | 4.6 % | 2.6 % | 1.0 % |
| 60S, B 60 | 6.0 % | 3.1 % | 1.1 % |
| 60S, B 120 | 6.9 % | 3.6 % | 1.1 % |

The z-score divides each pixel's correlations by one pooled standard deviation, so

* a **particle's** z comes from its own hypothesis: it moves up or down by that hypothesis's
  `sqrt(e)` factor, ~0 on average (measured with this change: mean dz +0.02 to +0.07,
  sd 0.25 to 0.32);
* the **noise maximum** over ~41 M hypotheses per pixel is drawn preferentially from the
  hypotheses with `e > mean`: the null tail is fatter. Half to two-thirds of the null
  peaks come from the noisiest fifth of templates. The null-calibrated 1-FP cut sits
  0.12 (B 0) to 0.21 z (B 60, B 120) higher than with this change, and that costs
  recall, not the particles' z.

Null-calibrated thresholds are still *correct* for Panther's z (they are measured on
Panther's own nulls); the z-score is just less sharp than it can be. The nominal
(formula) threshold only matched Panther's nulls at B 0 because this variance spread
(x 2.2 more null peaks) happened to offset the correlation between trials (x 0.4 to 0.45
of the formula with this change).

## 2. What changes

`after_ctf` multiplies hypothesis `(d, n)` by `s[d, n] = a[n] * b[d]`:

| factor | one per | computed | from | applied to |
|---|---|---|---|---|
| `a[n]` | template row | once per run (`_Run.__init__`) | reference noise: white noise under the ring-averaged `CTF^2` of every searched plane of every micrograph in the config | the template weights `U * S` (`Template.scale_rows`) and every error table's rows (`scale_table`) |
| `b[d]` | CTF plane | per micrograph (`_Run.prepare`) | the micrograph's own noise curves, after `a` | the micrograph's CTF planes (so the image features of plane `d`) and its noise / signal curves |

```
a[n] = ( mean_d e_ref[d, n] / mean e_ref )^(-1/2)
b[d] = ( mean_n a[n]^2 e[d, n] / mean (a^2 e) )^(-1/2)
```

The mean of `e` over all hypotheses is unchanged, so the scale of the threshold test and of
the z-score stay where they were.

**Why separable.** An exact `s[d, n]` is not a row operation on either GEMM operand: the
planes share the template rows, and the CTF lives on the image side. It would need the
reduction kernel (`lean_irfft_stats.cu`) to multiply every `(plane, row)` spectrum, and
per-plane moment forms. The separable form leaves 1.0 % sd of `e` (0.5 % in sigma). By the
same tail model that predicts the original effect, that residual inflates the null by
~4 to 5 %, about 0.01 z.

**What does not change.** The correlation loop, GEMMs, inverse FFT over psi, reduction
kernel, closed-form moments (`moments.py`; they are built from the same, now scaled,
weights and features), stage planning and the auto threshold formula. With `before_ctf` no
new code runs (`tests/test_normalization.py::test_before_ctf_runs_no_new_code`).

**Gate consistency.** The stage-1 threshold test predicts each hypothesis's kept and missed
noise from the error tables. Their rows are scaled with the templates (`a^2` for the
template's own spectra, `a` for cross-spectra with the unscaled signal model), and the
micrograph's noise and signal curves by `b^2`, so `q_kept` and `q_perp` follow the scaled
correlations and the ratios `rho` and `lam` are unchanged
(`tests/test_normalization.py::test_scaled_table_scales_the_threshold_test_consistently`).
An earlier prototype that rescaled a copy of the decomposition left the tables unscaled,
which is why its gate dropped some null peaks near the cut.

**Which variance is equalized.** That of the final stage's kept correlation (its table's
`reconstruction_autocorr`), the correlation the reported z-scores come from. Leopard-EM
equalizes the full projection's energy; at rank 96 x 80 the two differ by the truncation
error (~6 % of the energy, nearly constant over templates).

## 3. Using it

```yaml
# search config (after_ctf is the default; this line is only needed to go back)
normalization: before_ctf   # Panther's original normalization, for reproducing old results
```

The run summary (`search_summary.json`) gains a `normalization` record per micrograph:
the relative sd of `e` before, with `a` only and after, the range of `a`, and `b`.

Thresholds calibrated on `before_ctf` nulls do **not** carry over: recalibrate on nulls
searched with `after_ctf`. Without nulls, the automatic (formula) threshold still works and
is now conservative: on the synthetic micrographs of section 5 (B 0) at that threshold, recall is
37.2 % vs 35.3 % before, with 0.42 instead of 1.00 false picks per micrograph (real
xenon_131: 399 picks vs 403). The full recall gain needs the null-calibrated threshold, which
sits 0.13 to 0.28 z lower.

## 4. Cost

Setup only. `a` needs one ring-averaged CTF per micrograph in the config (1.1 s for the
xenon_131 run on CPU, once); `b` one `(13 x F) @ (F x 12288)` product per micrograph
(< 0.1 s). The search loop is unchanged; measured wall time per 4k 13-plane search is in
section 5.

## 5. Validation on xenon_131

The run and analysis scripts live outside this repository. Standard 13-plane production config (stage 1 rank 48 x 40 fp16 gate, stage 2 96 x 80),
nside 32 x 256 psi, 4k xenon_131, 20 phase-randomised nulls + the real micrograph per mode, 60S
template unfiltered (B 0) and B-filtered (B 60, B 120). Same GPU type, same seeds for both modes.

**Hypothesis variance** (relative sd of `e` over 13 x 12,288 (plane, template) pairs, from the
gate's own noise model on the real micrograph):

| | before (as is) | rows only (`a`) | after (`a` x `b`) |
|---|---|---|---|
| B 0 | 4.6 % | 2.6 % | 1.0 % |
| B 60 | 6.0 % | 3.1 % | 1.1 % |
| B 120 | 6.9 % | 3.6 % | 1.1 % |

**Where the null peaks come from** (null peaks above t_nom - 0.5, split by best template / plane):

| | noisiest fifth of templates holds | share by plane vs plane variance (r) |
|---|---|---|
| B 0 before / after | 51 % / 18 % | 0.97 / -0.10 |
| B 60 before / after | 60 % / 18 % | 0.89 / 0.00 |
| B 120 before / after | 62 % / 11 % | 0.82 / -0.18 |

**Null FP / independent-trial formula at t_nom - 0.35, and the null-calibrated 1-FP cut**
(counted; gate-corrected with ungated searches of nulls s1-s3 where the production gate drops
null peaks):

| | c before | c after | cut - nominal, before | cut - nominal, after |
|---|---|---|---|---|
| B 0 | 0.80 | 0.37 | -0.01 | -0.13 |
| B 60 | 0.75 | 0.21 | -0.02 | -0.20 |
| B 120 | 0.57 | 0.14 | -0.07 | -0.28 |

Gate: with scaled tables the gate drops null peaks only below 7.44 to 7.67 (B 60 / 120, 3
nulls each); at the new cut it moves the counted threshold by 0.011 to 0.014 z (corrected).
At B 0 (1 null) it drops nothing above 7.56, so no correction (cut 7.73).

**Particles.** Mean dz (after - before) at real picks +0.02 to +0.03 (sd 0.25 to 0.32); at the
2,800 true particles of 12 synthetic micrographs with injected 60S particles +0.07 (sd 0.26).

**Recall** (synthetic micrographs, B 0, 13 planes; each mode thresholded on its own 20 nulls):

| false picks per micrograph | 0.5 | 1 | 2 | 5 | 10 |
|---|---|---|---|---|---|
| recall gain (points) | +4.1 | +5.7 | +5.1 | +4.6 | +4.7 |
| 90 % bootstrap | 2.6-6.6 | 3.5-6.5 | 3.9-6.1 | 3.7-5.5 | 4.0-5.4 |

At each mode's own counted null cut: recall 35.5 % -> 40.5 % (993 -> 1,134 particles) with
1.00 -> 0.92 realized false picks per micrograph; at BH q = 0.05: 41.0 % -> 45.9 %. (Rescaling
per template only, on a copy of the decomposition, gained 1.9 to 3.5 points on the same
micrographs.)

**Real xenon_131 picks** at each mode's own cut: 403 / 394 / 378 -> 404 / 395 / 378 at B 0 / 60
/ 120 (402, 391 and 372 shared); few of its particles sit near the cut.

**Cost.** Median wall time per 4k search (one GPU, first search of each run excluded): B 0 97 s
-> 83 s (stage 2 34 s -> 22 s, as fewer noise pixels pass the gate), B 60 62 -> 60 s, B 120
60 -> 63 s.

**Regression.** `before_ctf` reproduces the earlier runs of the same configs: identical null
counts (B 0: 243 peaks at t_nom - 0.35, as before) and real z maps equal except 1 to 2 of 12.8 M
pixels (none above z 7; GPU run-to-run noise).

**Leopard-EM** (13 planes, its own grid: 1.58 M orientations per plane vs Panther's 3.15 M, so
not a 1:1 comparison): c 0.78 / 0.23 / 0.11 at B 0 / 60 / 120. After the fix Panther's null
matches Leopard's at B 60 / 120 (0.21, 0.14) but sits at about half of Leopard's at B 0 (0.37),
which these runs do not explain (at 3 planes, B 0, Leopard matched per-template equal-variance
Panther).

## 6. Limitations and open points

* **Astigmatism.** `e` uses ring means of `CTF^2`, so the in-plane (psi) dependence an
  astigmatic CTF adds is not equalized (Leopard-EM normalizes every psi). xenon_131 has
  (u - v) / 2 = 181 A of astigmatism; the size of this term is not measured.
* **Defocus range.** `a` is one set per run, from the config's micrographs' average CTF.
  Over the 13 planes (2400 A) the orientation x defocus interaction it cannot follow is
  the 1.0 % residual above; datasets with a much wider defocus range may want per-group
  `a` (not implemented).
* **Exported maps.** The Leopard-format export divides correlations by the median
  correlation sd; with scaled hypotheses the exported MIP / scaled MIP shift slightly
  (the z-score map does not).
* **Seeded / local stages** use the same scaled weights; not separately validated.

## 7. Files

| file | change |
|---|---|
| `src/panther_em/search/normalization.py` | new: the factors, table scaling, `AfterCtfNormalization.apply`, reference noise |
| `src/panther_em/search/template.py` | `Template.scale_rows`, `Template.row_scale`: row factors on the resident weights |
| `src/panther_em/managers/search.py` | `SearchConfig.normalization`; `_Run` applies `a` at start-up and `b` per micrograph; summary record |
| `tests/test_normalization.py` | new: exact equalization of separable `e`; residual = interaction; table scaling vs threshold test; `q_kept == e`; per-micrograph apply; `scale_rows`; end-to-end synthetic search (same particle label, pose and plane; z within 10 %) |
| `docs/after_ctf_normalization.md` | this file |
