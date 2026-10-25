"""Small-sample historical zone matching. No synthetic data or trained ML model."""
from __future__ import annotations
import argparse, json, math
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
from map_coordinates import (
    COORDINATE_SCALE, MAP_MIN, MAP_WIDTH, MAP_HEIGHT, MAP_SHORT_SIDE,
    REFERENCE_WIDTH, REFERENCE_HEIGHT,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "historical_zone_data.csv"
EVALUATION_PATH = DATA_DIR / "evaluation_report.json"
PUBLIC_SOURCE_URL = "https://raw.githubusercontent.com/furkanalialtuntas/pubg-safe-zone-predictor/main/zone_data.csv"
MAP_MAX, MAP_NAME = MAP_WIDTH, "Erangel"
SOURCE_LABEL = "furkanalialtuntas/pubg-safe-zone-predictor zone_data.csv; license not stated"
NESTING_TOLERANCE_PX = 12.0
REQUIRED_COLUMNS = ["Match_ID", "Phase", "White_X", "White_Y", "White_R"]


def validate_zone(zone: dict[str, Any], *, require_circle_in_map: bool = False) -> dict[str, Any]:
    try:
        x, y, radius = float(zone["center_x"]), float(zone["center_y"]), float(zone["radius"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Historical zone has missing or non-numeric geometry") from exc
    if not np.isfinite([x, y, radius]).all():
        raise ValueError("Historical zone contains non-finite geometry")
    if not (MAP_MIN <= x <= MAP_WIDTH and MAP_MIN <= y <= MAP_HEIGHT):
        raise ValueError("Historical zone center is outside the map coordinate bounds")
    if not 0 < radius <= MAP_SHORT_SIDE:
        raise ValueError("Historical zone radius is outside the supported range")
    if require_circle_in_map and (x - radius < MAP_MIN or x + radius > MAP_WIDTH
                                  or y - radius < MAP_MIN or y + radius > MAP_HEIGHT):
        raise ValueError("Historical candidate circle extends outside the reference map bounds")
    return {**zone, "center_x": x, "center_y": y, "radius": radius}


def load_zone_data(path: str | Path = DATA_PATH, *, fetch_if_missing: bool = False) -> pd.DataFrame:
    """Load local source CSV, optionally reading the public source into memory if absent.

    Remote data is never written to disk. Local data always takes precedence.
    """
    path = Path(path)
    source_reference = f"{path.name}; schema/source: {PUBLIC_SOURCE_URL}"
    coordinate_system = "Canonical Erangel reference units: source reference raster pixels (1181x1133, origin top-left) uniformly scaled by 1000/1133; X, Y and radius share units"
    if path.is_file():
        frame = pd.read_csv(path)
    elif fetch_if_missing:
        try:
            request = Request(PUBLIC_SOURCE_URL, headers={"User-Agent": "BGMI-Zone-Rotation-Predictor/1.0"})
            with urlopen(request, timeout=12) as response:
                payload = response.read()
            frame = pd.read_csv(BytesIO(payload))
            source_reference = f"transient public source (not saved locally): {PUBLIC_SOURCE_URL}"
        except (OSError, URLError, ValueError) as exc:
            raise FileNotFoundError(
                f"No local historical CSV at {path} and the public source could not be fetched: {exc}. "
                "Place a permitted CSV at data/historical_zone_data.csv; synthetic fallback is disabled."
            ) from exc
    else:
        raise FileNotFoundError(f"Historical CSV not found: {path}. Place zone_data.csv here; synthetic fallback is disabled.")
    names = {str(col).strip().casefold(): col for col in frame.columns}
    missing = [col for col in REQUIRED_COLUMNS if col.casefold() not in names]
    if missing:
        raise ValueError("Historical CSV missing required columns: " + ", ".join(missing))
    frame = frame.rename(columns={names[col.casefold()]: col for col in REQUIRED_COLUMNS}).copy()
    frame = frame[REQUIRED_COLUMNS + [col for col in frame.columns if col not in REQUIRED_COLUMNS]]
    frame["Match_ID"] = frame["Match_ID"].astype("string").str.strip()
    if frame.Match_ID.isna().any() or frame.Match_ID.eq("").any():
        raise ValueError("Match_ID must be present on every observation")
    for col in REQUIRED_COLUMNS[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[REQUIRED_COLUMNS[1:]].isna().any().any():
        raise ValueError("Phase and White_X/White_Y/White_R must be numeric and complete")
    if not (frame.Phase.between(1, 8) & (frame.Phase % 1 == 0)).all():
        raise ValueError("Phase must be an integer from 1 to 8")
    if not frame.White_X.between(0, REFERENCE_WIDTH).all():
        raise ValueError(f"White_X must fit source map width [0, {REFERENCE_WIDTH:g}]")
    if not frame.White_Y.between(0, REFERENCE_HEIGHT).all():
        raise ValueError(f"White_Y must fit source map height [0, {REFERENCE_HEIGHT:g}]")
    if ((frame.White_R <= 0) | (frame.White_R > min(REFERENCE_WIDTH, REFERENCE_HEIGHT))).any():
        raise ValueError("White_R must be positive and no larger than source map short side")
    if frame.duplicated(["Match_ID", "Phase"]).any():
        raise ValueError("Duplicate Match_ID/Phase observations are ambiguous")
    for match_id, match in frame.groupby("Match_ID", sort=False):
        phases = match.Phase.astype(int).tolist()
        if phases != sorted(phases):
            raise ValueError(f"{match_id}: phase rows are out of order")
        radii = match.White_R.to_numpy(float)
        if len(radii) > 1 and not np.all(np.diff(radii) < 0):
            raise ValueError(f"{match_id}: safe-zone radius must decrease at every observed phase")
    frame["Phase"] = frame.Phase.astype(int)
    frame = frame.sort_values(["Match_ID", "Phase"], kind="stable").reset_index(drop=True)
    frame["game"] = frame["game"].fillna("PUBG per source project") if "game" in frame else "PUBG per source project"
    frame["platform"] = frame["platform"].fillna("Not identified in source") if "platform" in frame else "Not identified in source"
    frame["map_name"] = frame["map_name"].fillna(MAP_NAME) if "map_name" in frame else MAP_NAME
    frame["source_reference"] = frame["source_reference"].fillna(source_reference) if "source_reference" in frame else source_reference
    frame["coordinate_system"] = frame["coordinate_system"].fillna(coordinate_system) if "coordinate_system" in frame else coordinate_system
    # Canonical application coordinates are the original reference-map pixels.
    # Never normalize axes independently: circles and transition geometry stay Euclidean.
    frame["center_x"] = frame.White_X.astype(float) * COORDINATE_SCALE
    frame["center_y"] = frame.White_Y.astype(float) * COORDINATE_SCALE
    frame["radius"] = frame.White_R.astype(float) * COORDINATE_SCALE
    frame["observation_validation_status"] = "valid_source_observation"
    return frame


def build_transitions(observations: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Pair adjacent phases within each match; flag pairs violating circle nesting."""
    rows, rejected = [], []
    for match_id, group in observations.groupby("Match_ID", sort=False):
        records = list(group.sort_values("Phase").to_dict("records"))
        for current, nxt in zip(records, records[1:]):
            phase, next_phase = int(current["Phase"]), int(nxt["Phase"])
            base = {"match_id": str(match_id), "phase": phase, "zone_number": phase, "next_phase": next_phase}
            if next_phase != phase + 1:
                rejected.append({**base, "reason": "non_consecutive_phase"})
                continue
            current_radius = float(current["White_R"])
            next_radius = float(nxt["White_R"])
            movement = math.hypot(float(current["White_X"] - nxt["White_X"]),
                                  float(current["White_Y"] - nxt["White_Y"]))
            containment_excess = movement + next_radius - current_radius
            if next_radius >= current_radius:
                rejected.append({**base, "reason": "radius_did_not_shrink",
                                 "current_radius_px": current_radius, "next_radius_px": next_radius})
                continue
            next_x_px, next_y_px = float(nxt["White_X"]), float(nxt["White_Y"])
            if (next_x_px - next_radius < 0 or next_x_px + next_radius > REFERENCE_WIDTH
                    or next_y_px - next_radius < 0 or next_y_px + next_radius > REFERENCE_HEIGHT):
                rejected.append({**base, "reason": "next_circle_out_of_bounds",
                                 "next_x_px": next_x_px, "next_y_px": next_y_px,
                                 "next_radius_px": next_radius})
                continue
            if containment_excess > NESTING_TOLERANCE_PX:
                rejected.append({**base, "reason": "next_circle_not_contained_in_current",
                                 "center_distance_px": movement, "current_radius_px": current_radius,
                                 "next_radius_px": next_radius, "containment_excess_px": containment_excess})
                continue
            rows.append({
                "match_id": str(match_id), "map_name": MAP_NAME, "game": str(current["game"]),
                "platform": str(current["platform"]), "data_source": SOURCE_LABEL,
                "source_reference": str(current["source_reference"]),
                "coordinate_system": str(current["coordinate_system"]), "zone_number": phase,
                "current_x": float(current["center_x"]), "current_y": float(current["center_y"]),
                "current_radius": float(current["radius"]), "next_x": float(nxt["center_x"]),
                "next_y": float(nxt["center_y"]), "next_radius": float(nxt["radius"]),
                "phase": phase, "next_phase": next_phase,
                "source_current_x_px": float(current["White_X"]), "source_current_y_px": float(current["White_Y"]),
                "source_current_radius_px": float(current["White_R"]), "source_next_x_px": float(nxt["White_X"]),
                "source_next_y_px": float(nxt["White_Y"]), "source_next_radius_px": float(nxt["White_R"]),
            })
    columns = ["match_id", "map_name", "game", "platform", "data_source", "source_reference",
               "coordinate_system", "zone_number", "phase", "next_phase",
               "current_x", "current_y", "current_radius", "next_x", "next_y", "next_radius",
               "source_current_x_px", "source_current_y_px",
               "source_current_radius_px", "source_next_x_px", "source_next_y_px", "source_next_radius_px"]
    return pd.DataFrame(rows, columns=columns), rejected


def predict_next_zone(transitions: pd.DataFrame, *, zone_number: int, current_x: float,
                      current_y: float, current_radius: float, minimum_matches: int = 2,
                      candidate_limit: int = 3) -> dict[str, Any]:
    """Return exact observed next circles from the nearest same-phase analogues."""
    query = validate_zone({"center_x": current_x, "center_y": current_y, "radius": current_radius})
    if not 1 <= int(zone_number) <= 8:
        raise ValueError("zone_number must be between 1 and 8")
    same_phase = transitions.loc[
        (transitions.zone_number.astype(int) == int(zone_number))
        & (transitions.map_name == MAP_NAME)
    ].copy()
    comparable_count = int(same_phase.match_id.nunique())
    if comparable_count < minimum_matches:
        raise ValueError(f"Only {comparable_count} comparable historical matches for phase {zone_number}; at least {minimum_matches} required.")
    same_phase["_distance"] = np.sqrt(
        ((same_phase.current_x.astype(float) - query["center_x"]) / MAP_SHORT_SIDE) ** 2
        + ((same_phase.current_y.astype(float) - query["center_y"]) / MAP_SHORT_SIDE) ** 2
        + ((same_phase.current_radius.astype(float) - query["radius"]) / MAP_SHORT_SIDE) ** 2
    )
    ranked = []
    for row in same_phase.sort_values(["_distance", "match_id"], kind="stable").drop_duplicates("match_id").head(candidate_limit).to_dict("records"):
        outcome = validate_zone({"center_x": row["next_x"], "center_y": row["next_y"],
                                 "radius": row["next_radius"]}, require_circle_in_map=True)
        ranked.append({**outcome, "source_match_id": str(row["match_id"]),
                       "source_phase": int(row["zone_number"]),
                       "source_next_phase": int(row.get("next_phase", int(row["zone_number"]) + 1)),
                       "similarity_distance": float(row["_distance"]),
                       "source_reference": str(row["source_reference"])})
    if not ranked:
        raise ValueError(f"No validated historical outcomes are available for phase {zone_number}.")
    return {
        **ranked[0], "model_name": "nearest historical match",
        "data_source": str(same_phase.iloc[0]["data_source"]),
        "game": str(same_phase.iloc[0]["game"]), "platform": str(same_phase.iloc[0]["platform"]),
        "comparable_matches": comparable_count, "candidates": ranked,
        "contributing_matches": [{
            "match_id": ranked[0]["source_match_id"], "phase": int(zone_number),
            "next_phase": int(zone_number) + 1,
        }],
        "estimate_match_count": 1,
        "matching_method": "single nearest same-phase observed transition; no averaging or coordinate synthesis",
        "feature_normalization": f"X, Y and radius differences divided by {MAP_SHORT_SIDE:g} canonical reference units (1000 per 1133 source pixels)",
        "note": "Source-reported PUBG Erangel outcome; platform and tournament provenance unverified; not verified BGMI data.",
    }


def evaluate_historical_matches(observations: pd.DataFrame, transitions: pd.DataFrame,
                                rejected: list[dict[str, Any]]) -> dict[str, Any]:
    """Leave out each complete match before retrieving nearest historical analogues."""
    actual, predicted, evaluated_matches, skipped = [], [], set(), 0
    match_ids = sorted(observations.Match_ID.astype(str).unique())
    for held_out in match_ids:
        test = transitions.loc[transitions.match_id == held_out]
        training = transitions.loc[(transitions.match_id != held_out)
                                   & (transitions.map_name == MAP_NAME)]
        had_prediction = False
        for row in test.to_dict("records"):
            try:
                estimate = predict_next_zone(training, zone_number=int(row["zone_number"]),
                    current_x=float(row["current_x"]), current_y=float(row["current_y"]),
                    current_radius=float(row["current_radius"]), minimum_matches=1, candidate_limit=1)
            except ValueError:
                skipped += 1
                continue
            actual.append([row["next_x"], row["next_y"], row["next_radius"]])
            predicted.append([estimate["center_x"], estimate["center_y"], estimate["radius"]])
            had_prediction = True
        if had_prediction:
            evaluated_matches.add(held_out)
    report = {
        "source": SOURCE_LABEL, "source_reference": PUBLIC_SOURCE_URL,
        "source_license": "No license found in inspected repository listing; source CSV is not bundled.",
        "game": "PUBG per project README; platform/date/tournament provenance not stated; BGMI unverified.",
        "source_matches": len(match_ids), "source_observations": int(len(observations)),
        "possible_adjacent_transitions": max(0, len(observations) - observations.Match_ID.nunique()),
        "usable_transitions": int(len(transitions)), "rejected_transitions": len(rejected),
        "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
        "rejected_transition_records": rejected,
        "evaluation": {
            "method": "leave-one-match-out; held-out match excluded entirely from analog search",
            "held_out_matches_with_predictions": len(evaluated_matches),
            "held_out_matches_total": len(match_ids), "held_out_transitions": len(actual),
            "skipped_transitions_without_comparable_phase": skipped, "metrics": None,
            "limitation": "Six matches and at most 18 transitions are insufficient to claim reliable BGMI prediction.",
        },
    }
    if actual:
        delta = np.asarray(predicted, float) - np.asarray(actual, float)
        center_error = np.hypot(delta[:, 0], delta[:, 1])
        report["evaluation"]["metrics"] = {
            "mae_next_x_canonical_units": float(np.abs(delta[:, 0]).mean()),
            "mae_next_y_canonical_units": float(np.abs(delta[:, 1]).mean()),
            "euclidean_center_error_canonical_units": float(center_error.mean()),
            "mae_next_radius_canonical_units": float(np.abs(delta[:, 2]).mean()),
            "mae_next_x_reference_pixels": float(np.abs(delta[:, 0]).mean() / COORDINATE_SCALE),
            "mae_next_y_reference_pixels": float(np.abs(delta[:, 1]).mean() / COORDINATE_SCALE),
            "euclidean_center_error_reference_pixels": float(center_error.mean() / COORDINATE_SCALE),
            "mae_next_radius_reference_pixels": float(np.abs(delta[:, 2]).mean() / COORDINATE_SCALE),
        }
    return report



def export_processed_observations(observations: pd.DataFrame, transitions: pd.DataFrame,
                                  rejected: list[dict[str, Any]], path: str | Path) -> None:
    """Write all observed source rows plus canonical geometry and outgoing-pair status."""
    status = {
        (str(row["match_id"]), int(row["zone_number"])): "accepted"
        for row in transitions.to_dict("records")
    }
    status.update({
        (str(row["match_id"]), int(row["phase"])): f"excluded:{row['reason']}"
        for row in rejected
    })
    out = observations.copy()
    out["source_center_x_px"] = out["White_X"].astype(float)
    out["source_center_y_px"] = out["White_Y"].astype(float)
    out["source_radius_px"] = out["White_R"].astype(float)
    out["converted_center_x"] = out["center_x"].astype(float)
    out["converted_center_y"] = out["center_y"].astype(float)
    out["converted_radius"] = out["radius"].astype(float)
    out["validation_status"] = out.apply(
        lambda row: status.get((str(row["Match_ID"]), int(row["Phase"])), "no_consecutive_observation"),
        axis=1,
    )
    out["source_phase"] = out["Phase"].astype(int)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH,
                        help="local historical zone_data.csv; local file takes precedence")
    parser.add_argument("--fetch-public", action="store_true",
                        help="if --data is missing, read the stated public CSV into memory (never save or redistribute it)")
    parser.add_argument("--report", type=Path, default=EVALUATION_PATH)
    parser.add_argument("--output-csv", type=Path,
                        help="write the uniformly converted canonical observation table, preserving raw columns and provenance")
    parser.add_argument("--audit", type=Path,
                        help="write transition inclusion/exclusion audit JSON")
    args = parser.parse_args(argv)
    observations = load_zone_data(args.data, fetch_if_missing=args.fetch_public)
    transitions, rejected = build_transitions(observations)
    if args.output_csv:
        export_processed_observations(observations, transitions, rejected, args.output_csv)
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps({
            "source": SOURCE_LABEL, "source_reference": PUBLIC_SOURCE_URL,
            "coordinate_system": str(observations.iloc[0].coordinate_system),
            "tolerance_source_pixels": NESTING_TOLERANCE_PX,
            "observations": int(len(observations)),
            "candidate_transitions": int(len(transitions) + len(rejected)),
            "accepted_transitions": int(len(transitions)),
            "rejected_transitions": rejected,
        }, indent=2), encoding="utf-8")
    if transitions.empty:
        raise SystemExit("No geometrically valid adjacent transitions; predictions are unavailable.")
    report = evaluate_historical_matches(observations, transitions, rejected)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
