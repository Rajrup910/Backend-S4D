# README media

Builds the horizontal GIFs in `docs/assets/readme/` and audits the root `README.md` against `results/`.

| File | Job |
|---|---|
| `facts.py` | Every number the README and its big-type scenes use, read from `results/`. Nothing is hand-entered |
| `scenes.py` | The big-type scenes (1000 x 480) |
| `capture_scenes.py` | Records those scenes in headless Chrome |
| `capture_real.py` | Lifts charts, step strips, the abstention simulator and the fold machine out of `review2_infographic.html` and records each alone at README-legible size. The page's own data and animation code run unchanged |
| `reels.py` | Joins the recordings into GIFs, with transitions and a progress bar |
| `audit_readme.py` | Checks every decimal, key claim, link and GIF in the README. Exits 1 on any failure |

```bash
python scripts/readme_media/capture_scenes.py
python scripts/readme_media/capture_real.py
python scripts/readme_media/reels.py
python scripts/readme_media/audit_readme.py
```

Recording runs on a virtual clock (`CLOCK_JS` in `cdp.py`): the page's timers, `requestAnimationFrame` and CSS animations only advance when the next frame is requested, so frames are exactly 40 ms apart and a slow frame cannot stall the footage. The GIFs play at 25 fps.

Only `audit_readme.py` runs from a clean clone. `capture_real.py` needs the infographic, which is kept locally and is not in the repository. All three capture steps need Chrome and network access for the page fonts. Frames go to `README_MEDIA_TMP`, or to the system temp folder if it is unset; a full capture uses about 1 GB there.

**Deliberate differences from the infographic.** In the simulator, "Missed Malignancies" reads "Missed serious cases", because the set includes AKIEC, which is pre-malignant. The big-type calibration scene labels ECE (soft vote) and Macro-F1 / missed cases (TTA stack) separately, because those numbers come from different stacks. Panels that S84 made out of date are not recorded.
