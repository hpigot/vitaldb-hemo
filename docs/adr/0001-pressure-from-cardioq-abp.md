# 1. Pair the flow with CardioQ/ABP, not SNUADC/ART

Date: 2026-10-29

## Context

The arterial pressure is recorded twice in the CardioQ cases: directly by the SNUADC acquisition device as `SNUADC/ART` at 500 Hz, and relayed through the CardioQ monitor as `CardioQ/ABP` at 180 Hz. The flow, `CardioQ/FLOW`, is recorded by the CardioQ monitor.

Measured from beat onsets, the lag from `CardioQ/FLOW` to `CardioQ/ABP` is steady within a case (40-120 ms between patients, interquartile range 5-55 ms within one). The lag from `CardioQ/FLOW` to `SNUADC/ART` jumps by several hundred milliseconds within minutes: the two devices keep separate clocks. A model fitted to flow and pressure misaligned by that much is wrong in a way the fit cannot show.

## Decision

`cardioq.load` returns `CardioQ/ABP` as the pressure. Cases where `CardioQ/ABP` is all zero (1018, 3311: the arterial line was not connected to the monitor) are excluded, in `cardioq.EXCLUDED`.

## Consequences

- Pressure and flow share one clock, and the remaining delay is physiological: pulse transit from the descending aorta to the wrist, plus any processing lag. It is reported per window.
- The pressure is at 180 Hz rather than 500 Hz. That is enough for beat onsets and the Windkessel fits, but loses detail in the upstroke and the dicrotic notch.
- Using `SNUADC/ART` with the flow would need alignment in every window, which cannot be separated from a change in pulse transit time. It is left as a possible extension, and it would bring back cases 1018 and 3311.
