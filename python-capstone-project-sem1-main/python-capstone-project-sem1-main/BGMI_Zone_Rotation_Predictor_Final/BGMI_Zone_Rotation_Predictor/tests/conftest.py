import pandas as pd
import pytest

# Deterministic TEST-ONLY points; these are not source match data and are never
# packaged under data/ or used for application predictions.
@pytest.fixture
def local_zone_csv(tmp_path):
    rows = []
    centers = [(500, 500), (600, 400), (400, 600), (300, 300)]
    for index, (x, y) in enumerate(centers, start=1):
        for phase, dx, dy, radius in (
            (1, 0, 0, 300), (2, 20, 10, 160),
            (3, 30, 15, 100), (4, 35, 20, 50),
        ):
            rows.append({
                "Match_ID": f"TEST-ONLY-{index}", "Phase": phase,
                "White_X": x + dx, "White_Y": y + dy, "White_R": radius,
            })
    path = tmp_path / "test_only_zone_data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
