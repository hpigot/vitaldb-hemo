# 2. Scale the flow once per case to the device stroke volume

Date: 2026-10-29

## Context

`CardioQ/FLOW` is a Doppler velocity signal in device units. The monitor turns velocity into stroke volume (`CardioQ/SV`) with an aortic cross-section from a nomogram of age, height and weight, so the ratio of `CardioQ/SV` to `CardioQ/SD` (stroke distance) is constant for a patient. The signal itself has no stated unit or scale, and the models need flow in mL/s.

## Decision

For each usable 10 s window, the gain is the device flow (mean `CardioQ/SV` times mean `CardioQ/HR` / 60) divided by the mean of `CardioQ/FLOW` over the whole beats in the window. The case gain is the median over windows, and `cardioq.load` returns `q = gain * FLOW` in mL/s. The column `ratio` keeps each window's gain over the case gain, and windows with a ratio more than 20 % from 1 are rejected (`reason == "gain"`).

## Consequences

- One number per case, so the shape and the beat-to-beat changes of the flow are the CardioQ's own; only the scale comes from the device stroke volume.
- The scale is that of the device stroke volume, total stroke volume by the nomogram, not descending-aortic flow alone. Any method calibrated on this flow inherits the device's scale, so absolute agreement with `CardioQ/SV` across patients says little; agreement over time within a patient is what can be tested.
- The window ratio drifts by up to 20 % over hours in some cases, for example as the probe moves. A per-window gain would follow that, but would also force every window's mean flow to match the device and hide real disagreement.
