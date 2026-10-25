# Historical zone data

historical_zone_data.csv is a locally generated processed observation table. The raw input is retained locally at source_zone_observations.csv, and the ZIP excludes both raw and processed CSV because the upstream repository does not state data usage terms.

The input format is Match_ID,Phase,White_X,White_Y,White_R. Reproducible processing, when you have permission to use the source data:

python train_model.py --data path/to/source_zone_data.csv --output-csv data/historical_zone_data.csv --audit data/transition_audit.json --report data/evaluation_report.json

All observed rows and original White_* source values are preserved in the processed table. converted_center_x, converted_center_y, and converted_radius use the canonical reference units: original source pixels uniformly multiplied by 1000/1133. This same conversion is applied to X, Y and radius, preserving circle geometry. Rows retain game/platform provenance, source reference, coordinate-system description and validation status. Platform and tournament provenance are not identified by the source; no BGMI attribution is made.

transition_audit.json lists all rejected adjacent phase pairs by match ID, phase, next phase and reason. It applies exact circle containment with a 12-source-pixel tolerance and never edits source geometry. The checked source version has 13 accepted and five rejected transitions out of 18 candidates.

The app prefers the local processed file. Without it, it may read the raw CSV from the stated public URL into memory and does not save it. This requires internet access.
