# Default detection threshold: 5 % false discovery rate (`fdr: 0.05`), analytic null

**Status:** the default. `fdr: null` restores the fixed threshold of one expected false
detection per micrograph. It builds on the after-CTF normalization
(`docs/after_ctf_normalization.md`); the validation below uses both.

> **Behaviour change.** Panther used to accept every pixel above one z threshold per
> search, set so that one false detection per micrograph is expected under the
> independent-trial formula. Now each micrograph's picks are the peaks above its own
> Benjamini-Hochberg threshold at FDR 5 %, under its own analytic null. A micrograph with
> many particles gets a lower cut (more picks); one with few or none gets a stricter cut
> (fewer false picks). The search also takes 1.6 to 1.9 times as long, plus about a minute
> per micrograph for the null (see Cost and Validation). `fdr_null: formula` uses the
> independent-trial count instead (about half the requested FDR).

## Rule

After the search, per micrograph (`panther_em.managers.fdr`):

1. **Peaks:** local maxima at least `fdr_peak_radius` (10) pixels apart among the ACCEPTED
   pixels, found with the particle export's peak finder, so `k` counts what the export
   writes.
2. **Null:** the expected number of false peaks above `t`. By default (`fdr_null:
   analytic`) it is the micrograph's own analytic null, which the search computes from its
   template, filters, grid and the micrograph's power spectrum (below). With
   `fdr_null: formula` it is the independent-trial count `λ(t) = N_positions ×
   N_hypotheses × Q(t)`, the same count that sets the 1-FP threshold. `N_positions` is the
   micrograph's searched positions (under its mask, if it has one).
3. **Threshold:** with peak z sorted high to low, the cut is `z_(k)` for the largest `k`
   with `λ(z_(k)) ≤ 0.05 k` (BH step-up, π0 = 1). No such `k` means no picks. The rule is
   Leopard-EM's engine-agnostic `leopard_em.analysis.fdr.bh_cutoff`.
4. **Relabel:** ACCEPTED pixels below the cut become REJECTED; ABSORBED pixels no longer
   within the final stage's `absorb_radius` of an ACCEPTED pixel become AMBIGUOUS. The run
   summary (`search_summary.json`) gets an `fdr` record per micrograph: the BH threshold,
   the applied threshold and floor, whether it was capped and whether the floor limited
   it (`floor_limited`), peaks considered and picked, the null used and where it anchored
   the thresholds, and the expected false picks at the cut under it and under the
   formula.

**Why not the formula.** Panther's trials are correlated, so the formula over-counts false
peaks: with after-CTF normalization the measured null on xenon_131 is 0.37 (B 0) to 0.14
(B 120) of it, and the realised FDR comes out at about half of `q`. The analytic null
(`docs/null_calibration.md`) gives the cut the FDR asks for: on the 12 Track S micrographs
the realised FDR at `q` = 5 % is 4.5 % (formula: 2.0 %) and recall 46.0 % (43.6 %).

**The analytic null (`fdr_null: analytic`, default).** Before its search, each micrograph's
null is predicted with `panther_em.managers.null.predict_search_null`, the same code as
`panther-em null`'s analytic mode, over its searched positions. It takes about a minute
for a 13-plane 4k micrograph on a GPU (`fdr_analytic` sets the sampling, as `panther-em
null`'s `analytic`; the template volume is the one the decomposition records, or
`fdr_volume`). A search the model cannot cover (a final stage over a subset of
orientations, a non-HEALPix grid, several Fourier filters) keeps the formula, and its
record says why.

**A null from `panther-em null` (`fdr_null: <null_summary.json>`).** Each micrograph's λ is
then the summary's: its analytic prediction (`fdr_null_source: analytic`) or the
phase-randomised copies' rate table (`empirical`), scaled from the null's valid area to
the micrograph's searched positions. A micrograph the summary lacks keeps the formula,
and its record says so. On xenon_131's Leopard-EM searches, the same rule with the analytic
null picks about 30 more particles at 5 % than the 1-FP threshold does (Leopard-EM's
validation). For thresholds on z after Leopard-EM's refinement, use the summary's
`analytic_refined` prediction with `leopard_em.analysis.bh_cutoff` on the refined z.

## Reach: the stages must look below the old threshold

The cut is only known after the search, but the stages decide per pixel during it. With
`fdr` set:

| | Default | Effect |
|---|---|---|
| `fdr_reach` | 0.5 | The final stage's `auto` threshold sits 0.5 z below the 1-FP threshold. It is the lowest z the FDR cut can take (the floor). |
| `fdr_gate_reach` | 0.25 | Earlier stages' `auto` thresholds (the stage-1 gate) are lowered by 0.25 z. The gate keeps pixels about 0.3 z below its own threshold, so this covers the 0.5 floor. |
| `fdr_anchor` | `null` | The 1-FP threshold both are measured from: the micrograph's null's own 1-FP cut. Every `auto` threshold moves by that cut minus the formula's. `formula`: the independent-trial 1-FP threshold, the same at every B (the behaviour before this option). |

**Why the anchor.** The formula's 1-FP threshold ignores the template, but the null, and
with it the FDR cut, falls as the field gets smoother. On the 12 synthetic micrographs the
null's 1-FP cut is 7.78, 7.67 and 7.58 at B 0, 60 and 120, against the formula's 7.86. With
the reach measured from the formula, the floor stays at 7.36 at every B. At B 120 it sat
0.03 to 0.12 z below the cuts on the dense micrographs, and on one micrograph the rule
picked every peak the search kept. Measured from the null, the floor is 7.28, 7.17 and
7.08.

**Measured on the 12 synthetic micrographs** (after-CTF, `fdr_null: analytic`, q = 5 %; same searches
otherwise, the prototype with self-seeded stage 1). For each B, the first row measures the reach from the
formula and the second from the null:

| B | Anchor | Floor | Recall | Realised FDR (false / expected) | Stage-2 pixels | Search time | Peak GPU memory |
|---|---|---|---|---|---|---|---|
| 0 | formula | 7.36 | 46.0 % | 4.5 % (61 / 63.1) | 135–147 k | 130–141 s | 31.2–31.4 GB |
| 0 | null | 7.28 | 46.1 % | 4.7 % (64 / 63.7) | 204–218 k | 162–169 s | 31.2–31.5 GB |
| 60 | formula | 7.36 | 36.6 % | 5.0 % (54 / 48.3) | 2.2–3.2 k | 67–75 s | 31.2–31.4 GB |
| 60 | null | 7.17 | 37.6 % | 4.9 % (54 / 49.8) | 6.6–8.5 k | 72–76 s | 31.2–31.5 GB |
| 120 | formula | 7.36 | 25.8 % | 4.5 % (34 / 32.3) | 0.3–0.8 k | 67–69 s | 31.2–31.4 GB |
| 120 | null | 7.08 | 27.9 % | 5.4 % (45 / 38.7) | 1.5–2.5 k | 69 s | 31.2–31.5 GB |

The time and memory columns cover three micrographs with each anchor. Measured from the formula, the floor
limited the cut on one micrograph at B 120. Measured from the null, it limited none. Recall rises by 1.0
point at B 60 and 2.1 at B 120. At B 0 the anchor moves the floor by only 0.08 z, but it adds about 20 % to
the search time (stage 2 sees 1.5× the pixels). Peak memory is set by stage 1 and does not change.

**Seeing the floor.** A BH threshold below the floor is clamped (`capped`). The search keeps
no peaks below the floor, so a cut the floor stops usually shows instead as every peak
picked. `floor_limited` flags either case: a lower floor might have picked more.

The gate is what costs time: every pixel it passes gets the exact stage-2 search.

**Cost** on 4k xenon_131, 13 planes, B 0, one GPU:

| Stage-1 gate | Stage-2 pixels | Stage-2 time | Total |
|---|---|---|---|
| at the 1-FP threshold (`fdr: null`) | 47 k | 26 s | 93 s |
| 0.25 z lower (this default) | 148 k | 84 s | 178 s |
| 0.7 z lower | 1.07 M | 505 s | 573 s |

A gate lowered 0.4 z ran out of memory on a 48 GB GPU with the default store settings. The
0.25 default does not.

Only `"auto"` thresholds are affected. A stage with a numeric threshold keeps it. If the
final stage's threshold is numeric, the FDR rule is off.

## Validation

The run and analysis scripts live outside this repository. Setup: xenon_131's 13-plane
production config, B 0, after-CTF normalization.

**Synthetic micrographs:** 12 4k micrographs on xenon_131's noise spectrum with 2,800
injected 60S particles (300 in 8 micrographs, 150 in 2, 50 in 2). A pick is true within
10 px of an injected particle.

| | Threshold | Recall | False picks (per micrograph) | Realised FDR |
|---|---|---|---|---|
| `before_ctf`, 1-FP threshold (Panther before both changes) | 7.861 | 35.3 % | 12 (1.0) | 1.2 % |
| `after_ctf`, 1-FP threshold (`fdr: null`) | 7.861 | 37.2 % | 5 (0.4) | 0.5 % |
| `after_ctf`, `fdr: 0.05` (the default) | 7.607 to 7.940 | **43.6 %** | 25 (2.1) | **2.0 %** |

- **The cut follows the particle count:**
  - 300 particles: 7.61 to 7.66.
  - 150 particles: 7.69 to 7.73.
  - 50 particles: 7.89 to 7.94, above the 1-FP threshold. Sparse micrographs get a stricter cut.

  None reached the 7.36 floor.
- **The realised FDR is below 5 %.** The formula expected 59 false picks; 25 were found, 0.42 of the estimate. The measured null at B 0 is 0.37 of the formula.
- **Lowering the gate adds 0.8 points of recall.** The same per-micrograph cuts, applied to searches whose gate sat at the 1-FP threshold, give 42.8 % recall with 22 false picks.

**Real xenon_131** (4k, about 400 ribosomes):

| | Threshold | Picks | Search time |
|---|---|---|---|
| `before_ctf`, 1-FP threshold | 7.861 | 411 | 97 s |
| `after_ctf`, 1-FP threshold | 7.861 | 408 | 83 s |
| `after_ctf`, `fdr: 0.05` | 7.471 (BH, not capped) | 434 | 137 s |

At 7.471 the formula expects 20.8 false picks (4.8 % of 434). Scaled by the measured null
ratio of 0.37, that is about 8 (1.8 %).

**Time.** The default takes about 1.6× as long as `fdr: null`:
- Synthetic micrographs: median 140 s (127 to 150) against 89 s. Stage 2 evaluates 146 k pixels instead of 41 k.
- Real micrograph: 137 s against 83 s.

The first two rows of each table count distinct peaks 10 px apart; the default's rows count
the export's picks, also 10 px apart.

## Using it

```yaml
# search config: the defaults
fdr: 0.05            # null: the old fixed 1-FP threshold
fdr_reach: 0.5       # floor of the FDR cut below the 1-FP threshold, z
fdr_gate_reach: 0.25 # how far the stage-1 gate is lowered, z
fdr_anchor: null     # reach measured from the null's own 1-FP cut, or formula
fdr_peak_radius: 10  # peak separation for counting, pixels
fdr_null: analytic   # or formula, or a null_summary.json of `panther-em null`
# fdr_null_source: analytic   # with a null_summary.json: or empirical (the copies)
# fdr_analytic: {n_mean_samples: 3000, n_kappa_pairs: 256, n_lattice_samples: 128}
```

`fdr: null` with `normalization: before_ctf` reproduces results from before both changes.

## Files

| File | Change |
|---|---|
| `src/panther_em/managers/fdr.py` | New: the formula null, the calibrated null from a `panther-em null` summary, the BH threshold (Leopard-EM's `bh_cutoff`), the relabelling; `FdrNull`, `null_cut_1fp`, `floor_limited` |
| `src/panther_em/managers/search.py` | `SearchConfig.fdr`, `fdr_reach`, `fdr_gate_reach`, `fdr_anchor`, `fdr_peak_radius`, `fdr_null`, `fdr_null_source`, `fdr_volume`, `fdr_analytic`; `_Run.n_hypotheses`, `fdr_active`, `prepare_fdr` (the null per micrograph, before the search, and its anchor), `apply_fdr`; lowered and anchored `auto` thresholds (masked micrographs included); `fdr` record and `fdr_null_s` in the summary |
| `src/panther_em/managers/null.py`, `_analytic.py` | `predict_search_null` and `search_volume` for the search; `AnalyticSpec` shared by both programs |
| `scripts/threshold_test_pipeline.py` | The search's `fdr_analytic` sampled as the null step's (fast on a CPU) |
| `tests/test_fdr.py` | New: the formula, BH step-up (the shared rule), no-pick case, the calibrated null (area scaling, missing micrograph), thresholds lowered and anchored only when active, the null's 1-FP cut, end-to-end search with the formula, a calibrated null and the analytic null (anchored on it, or on the formula) |
| `tests/test_managers.py` | `test_export_stack`: a lone particle's accepted region may now be its peak pixel only; `test_masked_search`: the `auto` thresholds are lowered by the reach and moved by the anchor, and the FDR null counts the searchable positions only; `test_search_resumes`: `fdr_null_s` in the timing |
| `docs/after_ctf_normalization.md` | Points to this file |
