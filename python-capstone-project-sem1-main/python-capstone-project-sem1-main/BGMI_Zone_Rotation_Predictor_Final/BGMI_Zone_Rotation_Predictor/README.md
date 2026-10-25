# BGMI Zone Rotation Predictor

The existing single-screen Streamlit application retains the uploaded Erangel map, draggable/resizable safe-zone overlay, team marker, squad and movement controls, route alternatives, and strategy panel.

## Historical source and limits

The upstream repository README reports six real PUBG Erangel matches and separately describes synthetic augmentation. This application reads only the raw five-column zone_data.csv; it does not use the synthetic rows or the upstream trained model. The source says PUBG, but platform, match dates, tournament provenance, and CSV license are not stated. This is not verified BGMI data. See the [source repository](https://github.com/furkanalialtuntas/pubg-safe-zone-predictor) and [source CSV](https://raw.githubusercontent.com/furkanalialtuntas/pubg-safe-zone-predictor/main/zone_data.csv).

The locally retained raw source file is data/source_zone_observations.csv and the reproducible processed output is data/historical_zone_data.csv; neither is included in the ZIP. The checked source version has 24 observations, six match IDs and four phases per ID. It has 18 possible consecutive same-match transitions. Thirteen pass circle containment; five are excluded, with match/phase/reason in data/transition_audit.json. The source CSV and processed observation CSV are not included in the ZIP because the repository states no data license. When no local data/historical_zone_data.csv exists, the app reads the raw public CSV into memory over HTTPS only; it never writes a download to disk. A local file takes precedence.

## Coordinates and map registration

The source reference raster is 1181×1133 pixels, origin at top left. Canonical app units are source-reference pixels multiplied uniformly by 1000/1133. Historical center X, center Y, and radius all use this same scale, so Euclidean distances and circles are preserved. The earlier importer divided X by 1181 and Y/radius by 1133, then treated all normalized units as equal. The app also mapped clicks independently over the 474×429 uploaded image and used its short side for radius. These incompatible scales distorted distances and caused historical outcomes to be compared against different map geometry.

The uploaded image is retained at its original 474×429 aspect ratio. First, every source coordinate (including radius) is multiplied by one uniform factor, 1000/1133, to form canonical units. The landmark-calibrated axis-aligned image transform is then:

- image_x = 0.47472603 × canonical_x + 0.72323129
- image_y = 0.46064945 × canonical_y − 6.69447652

The fit uses 22 named-location correspondences across the map. Their digitized residual is 15.32 image pixels RMS and 25.73 pixels maximum; three inconsistent text-label anchors (Shooting Range, Hospital, Primorsk) were excluded. That residual is a remaining map-registration limitation. Input clicks use the inverse transform. Zone centers, radius, team, predicted center, routes and destination all use canonical reference units. Radius is never independently rescaled in the predictor. The map component transforms points and samples the same source-coordinate circle for display, preserving its geometry through resize and zoom.

Importer checks require complete numeric fields, valid reference-map bounds, unique match/phase rows, phase order and shrinking observed radii. Transitions are formed only from consecutive phases within the same match. The containment check is distance(current_center, next_center) + next_radius <= current_radius + 12 source pixels; records are never altered to pass it. Candidate next circles must also fit inside the source reference raster.

## Matching, evidence and routes

The method is deterministic nearest-history matching. It compares same-phase current centers and radii in canonical reference units, normalizing all three feature differences by 1000 canonical units (the 1133-pixel source-map short side after uniform scaling). Each match can contribute at most one transition for a phase. The estimate is the exact next circle from the single nearest eligible recorded match; the app does not average observations or synthesize a circle. It identifies that match and phase and lists other distinct-match analogues. If comparable evidence is insufficient, the prediction remains unavailable.

Route alternatives are planned only inside the single reported historical estimate, so the recommended destination cannot silently belong to a different candidate circle. Routes outside the reference map or beyond the configured time deadline are discarded. Travel time uses assumed speeds and straight-line distance; terrain, roads, cover and live opponents are not modeled.

Example from the checked source: querying phase 3 at match mac05's observed center (466, 291), radius 101, returns mac05's observed phase 4 circle (458, 263), radius 57 source pixels (canonical center/radius are uniformly scaled by 1000/1133). The valid phase-3 evidence comes from mac05, mac01 and mac06; the estimate itself uses mac05 only. This is an illustration of matching, not a BGMI accuracy claim.

## Held-out evaluation

The evaluation holds out each complete match before searching for analogs. For this source version it evaluated 13 transitions across six held-out matches, with no held-out transitions skipped. Mean absolute error was 142.46 source pixels in X, 189.15 in Y, 253.27 pixels Euclidean center error, and 9.46 pixels in radius. Six matches are too few to establish dependable prediction performance; these metrics are descriptive only.

## Prepare a local processed CSV

The importer accepts the original columns Match_ID,Phase,White_X,White_Y,White_R. If you have permission to use a source CSV, run from the project folder:

python train_model.py --data path/to/source_zone_data.csv --output-csv data/historical_zone_data.csv --audit data/transition_audit.json --report data/evaluation_report.json

This writes all observed rows with original source fields, canonical converted geometry, provenance, coordinate system, and validation status. It does not repair, add, or invent observations. The original input CSV stays at its supplied path. The generated CSV is excluded from the downloadable ZIP unless data usage rights are established.

## Windows setup and launch

1. Extract the ZIP.
2. Double-click setup.bat once.
3. Double-click run.bat; the app opens at http://localhost:8501.

Restart from PowerShell in the extracted project folder with:

.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501

Run the tests and compilation checks with:

.\.venv\Scripts\python.exe -m pytest -q

.\.venv\Scripts\python.exe -m compileall -q app.py train_model.py map_coordinates.py tests

## Project files

- app.py: UI, strategy integration and route planning.
- train_model.py: source import, geometry validation, transitions, matching and evaluation.
- map_coordinates.py: canonical coordinate bounds and reference-to-uploaded-map landmark transform.
- assets/maps/erangel_user_upload.png: original uploaded map.
- assets/map_component/index.html: interactive map component and overlays.
- data/source_zone_observations.csv and data/historical_zone_data.csv: local raw and processed rows; both omitted from ZIP due unestablished data usage rights.
- data/transition_audit.json: exact rejected transition details.
- map_coordinates.py: registration and coordinate conversion constants, including the landmark table.
- data/evaluation_report.json: source counts, rejected match-phase pairs and match-level holdout metrics.
- data/map_registration_report.json: landmark mapping coefficients, anchors and residuals.
- tests/: importer, geometry, registration, matching, strategy and route tests.
