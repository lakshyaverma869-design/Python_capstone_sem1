from app import save_run, list_runs

def test_prediction_history_round_trip(tmp_path):
    path=tmp_path/"history.sqlite"
    run_id=save_run({"zone":3},{"center_x":1,"center_y":2,"model_name":"test"},{"label":"Rotate"},path)
    runs=list_runs(path)
    assert runs[0]["id"]==run_id
    assert runs[0]["inputs"]=={"zone":3}
    assert runs[0]["prediction"]["model_name"]=="test"
