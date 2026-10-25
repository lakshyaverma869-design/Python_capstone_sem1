from app import plan_routes
from app import inside_circle

def test_routes_inside_zone_and_before_deadline():
    prediction={"center_x":500,"center_y":500,"radius":200}
    routes=plan_routes((500,500),prediction,100,walking_speed=4,movement_mode="walk")
    assert routes
    assert all(r["time_feasible"] for r in routes)
    assert all(inside_circle(r["destination"],(500,500),200) for r in routes)

def test_deadline_and_vehicle_constraints():
    prediction={"center_x":950,"center_y":950,"radius":20}
    assert plan_routes((0,0),prediction,1,vehicle_available=False)==[]
    assert plan_routes((0,0),prediction,1,vehicle_available=True,movement_mode="vehicle")==[]

def test_vehicle_route_available_when_feasible():
    prediction={"center_x":600,"center_y":500,"radius":100}
    routes=plan_routes((500,500),prediction,30,vehicle_available=True,movement_mode="auto")
    assert any(r["mode"]=="vehicle" for r in routes)

def test_movement_preference_and_vehicle_availability_are_respected():
    prediction={"center_x":600,"center_y":500,"radius":100}
    walking_routes=plan_routes(
        (500,500),prediction,120,vehicle_available=True,movement_mode="walk"
    )
    vehicle_routes=plan_routes(
        (500,500),prediction,120,vehicle_available=True,movement_mode="vehicle"
    )
    no_vehicle_routes=plan_routes(
        (500,500),prediction,120,vehicle_available=False,movement_mode="auto"
    )
    assert walking_routes and {route["mode"] for route in walking_routes} == {"walk"}
    assert vehicle_routes and {route["mode"] for route in vehicle_routes} == {"vehicle"}
    assert no_vehicle_routes and {route["mode"] for route in no_vehicle_routes} == {"walk"}

def test_score_weights_are_configurable_and_normalized():
    routes=plan_routes((500,500),{"center_x":600,"center_y":500,"radius":100},60,True,travel_weight=2,risk_weight=1)
    assert routes and routes[0]["weights"]=={"travel_time":2/3,"route_risk":1/3}
