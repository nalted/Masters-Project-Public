To train a model, run:

python model_train.py --model-out model_outputs/xgb_18x275_minQ2=1_29Jan2026.json --input-files rootfiles/train/18x275/minQ2\=1/*.root --beam-electrons 18 --beam-protons 275

This saves both:
- the model JSON you requested, and
- a matching validation file (for the example above): model_outputs/xgb_18x275_minQ2=1_29Jan2026_18x275_val.npz
- separate pre-split input plots in a campaign folder under pictures/:
	- pictures/train_18x275/xgb_18x275_minQ2=1_29Jan2026_18x275_input_E_over_p.png
	- pictures/train_18x275/xgb_18x275_minQ2=1_29Jan2026_18x275_input_isolation_fraction_scan.png
	- pictures/train_18x275/xgb_18x275_minQ2=1_29Jan2026_18x275_input_acoplanarity.png
	- pictures/train_18x275/xgb_18x275_minQ2=1_29Jan2026_18x275_input_leadingpt_pt_charge.png

Notes:
- Training outputs are automatically tagged with beam energies (for example, `..._18x275...`) unless `--campaign-tag` is provided.
- You can override the auto tag with `--campaign-tag your_label`.
- `E/p` input plotting is fixed to `[0.0, 1.5]` (neutral particles use `NaN`, handled natively by XGBoost).
- `pt` input plotting is fixed to `[0, 5]` GeV.

Optional isolation plotting controls during training:

python model_train.py --model-out model_outputs/xgb_18x275_minQ2=1_29Jan2026.json --input-files rootfiles/train/18x275/minQ2\=1/*.root --beam-electrons 18 --beam-protons 275 --isolation-cone-size 2.5 --isolation-cone-sizes 1.5 2.0 2.5

Notes:
- `--isolation-cone-size` sets the cone used for the training feature `isolation_frac`.
- `--isolation-cone-sizes` sets the list of cone sizes shown in the isolation-fraction scan plot.
- `--signal-generator-statuses` controls which generator-status values are treated as final-state when selecting the first scattered electron candidate (default: `1`).

To evaluate and make all report plots, run:

python model_evaluate.py --model-in model_outputs/xgb_18x275_minQ2=1_29Jan2026.json --val-data model_outputs/xgb_18x275_minQ2=1_29Jan2026_18x275_val.npz --output-prefix pictures/my_model_eval

Notes:
- Plot filenames are automatically tagged with beam energies (for example, `..._18x275_...`) unless `--campaign-tag` is provided.
- You can override the auto tag with `--campaign-tag your_label`.
- Evaluation plots are saved under `pictures/eval_<tag>/`.
- Plot titles follow publication-style formatting and include context prefixes (`Training:` or `Stress Test:`).
- Legends use `scattered electron` wording (instead of `signal`) for electron-class entries.

To run stress-test inference on stress ROOT campaigns (supports normal and underscore association branch names), run:

python stress_test.py --model-in model_outputs/xgb_18x275_minQ2=1_29Jan2026.json --input-files rootfiles/misc/stress/10x275/minQ2\=1/*.root --output-prefix model_outputs/stress_10x275_minQ2=1 --campaign-tag 10x275_minQ2=1 --val-data model_outputs/xgb_18x275_minQ2=1_29Jan2026_val.npz --isolation-cone-size 2.5 --signal-generator-statuses 1 --save-csv

Optional cone scan customization:

python stress_test.py --model-in model_outputs/xgb_18x275_minQ2=1_29Jan2026.json --input-files rootfiles/misc/stress/10x275/minQ2\=1/*.root --output-prefix model_outputs/stress_10x275_minQ2=1 --campaign-tag 10x275_minQ2=1 --isolation-cones 0.4 0.8 1.2 1.5 2.0 2.5 3.0 3.5 4.0

This writes:
- *_scores.npz with per-particle score arrays and features
- *_summary.txt with particle counts and metrics
- optional *_scores.csv when --save-csv is provided
- the same plot suite as `model_evaluate.py`, saved under `pictures/stress_<tag>/` (avg gain, total gain, purity-efficiency with max-F1 threshold, Q2-x map, TP/TN/FN input distributions)
- *_input_distributions_3class.png in `pictures/stress_<tag>/` with overlaid feature distributions for all particles, scattered electrons, and charged HFS
- *_isolation_cone_scan_3class.png in `pictures/stress_<tag>/` with the same three classes scanned across cone-size choices from `--isolation-cones`

Notes:
- Stress outputs are also tagged automatically with beam energies unless `--campaign-tag` is provided.
- If your `--output-prefix` already includes the selected tag, the tag is not appended again (prevents duplicate suffixes).
- `--val-data` is optional for stress inference; provide it when you want threshold selection from validation instead of stress labels.
- Use `--no-plots` if you only want score outputs.
- `--isolation-cone-size` sets the cone used for the model input feature `isolation_frac` during stress inference.
- `--signal-generator-statuses` controls which generator-status values are accepted when choosing the first final-state electron as truth signal (default: `1`).
- Stress-test figure titles include `Stress Test:` and use publication-style capitalization.
- Isolation scan figures are titled `Iso Frac for Different Cone Sizes`.

