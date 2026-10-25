from app import plan_routes
from app import recommend_strategy

def test_recommendation_uses_best_route_and_explains_assumptions():
    pred={"center_x":550,"center_y":500,"radius":200}
    routes=plan_routes((500,500),pred,120,True)
    result=recommend_strategy(routes,(500,500),pred,120,"PUBG historical observations")
    assert result["feasible"] and result["primary"]==routes[0]
    assert "PUBG historical analog estimate; not BGMI-specific" in result["explanation"]
    assert "terrain" in result["explanation"]

def test_no_route_means_no_action_recommended():
    result=recommend_strategy([], (0,0), {"center_x":1,"center_y":1,"radius":1}, 0)
    assert not result["feasible"] and result["label"]=="No feasible route"
