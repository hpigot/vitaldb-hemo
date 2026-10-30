# 5. Evaluate pressure-only estimates with one fixed calibration protocol

Date: 2026-10-30

## Context

A stroke volume estimate from pressure alone has no absolute scale, so it has to be calibrated against the device before it can be compared with it. Every choice in that comparison changes the numbers: how long the calibration is, whether its windows are left out, what is averaged before comparing, how cases are pooled, and over what interval changes are counted for the concordance. Methods are only comparable if these choices are the same, and each of them is easy to get subtly wrong, for example by evaluating on the calibration windows.

## Decision

`evaluate.calibrated(table, drive, minutes=15, step=5)` fixes one protocol. Per case, the scale `k` is the median of device SV over the drive across the windows that make up the first 15 minutes of usable time. Those windows are left out. The rest is averaged over whole minutes of case time, agreement is computed per case and pooled with repeated-measures limits, and the concordance uses every fifth minute mean. Windows where the drive is NaN are skipped, so a filter can warm up. The function also returns the minute means with `ln(est / sv)`, the change in MAP since calibration and the time since calibration, for plotting.

## Consequences

- An estimator is anything that produces one number per usable window that stroke volume is proportional to. Its author does not write calibration or evaluation code, and two estimators evaluated this way are directly comparable.
- The median makes `k` robust to a few bad windows, but a calibration period that is unrepresentative of the rest of the case still biases everything after it; that is a property of calibration, which the protocol shows rather than hides.
- 15 minutes of usable time can span much more clock time when the early part of a case has many rejected windows. The protocol counts usable time, because that is the time with flow available.
- Estimators that are calibrated in another way, for example a filter that keeps adapting while flow is available, do not fit the protocol and need their own evaluation code built from `agreement` and `concordance`.
