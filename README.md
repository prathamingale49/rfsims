# RF Sims

Interactive RF link-budget and LoRa trade-study tool built with Streamlit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run rf.py
```

The app updates calculations and plots immediately when an input changes.

## Current features

- Editable TX/RX link-budget inputs
- VSWR-to-mismatch-loss calculation
- EIRP, FSPL, received power, and simple link margin
- Saved in-session band profiles with JSON import/export
- Default 915 MHz and 2.4 GHz profiles
- Distance sweeps for received power and link margin
- Free-space path-loss sweep versus frequency
- Editable LoRa mode table with SF, bandwidth, coding rate, required SNR, and optional sensitivity override
- Estimated LoRa sensitivity from `kTB + NF + required SNR` when no sensitivity override is supplied
- Highest-throughput viable LoRa mode versus distance
- Rough crossover/swap distance between a preferred near-range band and fallback band

## Important modeling notes

The default LoRa SNR thresholds and 2.4 GHz hardware values are placeholders. Replace them with LR2021 datasheet values or measured results before treating the adaptive-mode output as a design result.

The propagation model currently uses free-space path loss plus explicit user-entered losses. Ground reflection, terrain, airframe shadowing, fading, atmosphere, and measured installed antenna patterns should be added as separate models as the tool matures.
