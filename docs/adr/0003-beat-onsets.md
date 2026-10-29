# 3. Detect beat onsets with an own implementation of Zong et al. (2003)

Date: 2026-10-29

## Context

Beat onsets are needed for the per-beat features, for pairing pressure and flow beats, and for the delay between them. The usual method for arterial pressure is Zong et al. (2003): a 16 Hz low-pass filter, a slope sum function over 128 ms and an adaptive threshold. BioSPPy implements it, but version 2.2.4 fails to import (a missing `peakutils` dependency) and pulls in opencv, scikit-learn and h5py.

## Decision

`beats.onsets` implements the method in numpy and scipy, with one change: the onset is the last sample within 2 % of the upstroke height above the minimum in the 150 ms before the threshold crossing, rather than where the slope sum starts rising. The paper's rule is dragged up to 100 ms early by noise in diastole, and on the flow, whose diastole is flat, it lands at the start of the flat stretch.

## Consequences

- No heavy dependency, and the same function works on pressure and on flow.
- Onsets are placed on the low-pass signal, so they are smeared by about 15 ms on sharp corners. The window delays are accurate to about 20 ms.
- Systolic area and other features that need the end of ejection are not provided; they would need a dicrotic-notch detector.
