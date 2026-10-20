import math
import pytest
from autolife_shelf_grasp_306.navigation_map import world_to_pixel,pixel_to_world,manual_pose
from autolife_shelf_grasp_306.navigation_logic import navigation_summary


def test_target_failure_is_not_hidden_by_ready_pose():
    d=dict(ready=True,pose=[1,2,3],navigation=dict(state='failed',message='入住区等待目标可达超时'))
    assert '桌柜' in navigation_summary(d)
    d['localization_pending']=True
    assert '等待本轮定位确认' in navigation_summary(d)


@pytest.mark.parametrize('yaw',[0,.7,-1.2,math.pi])
def test_map_round_trip_with_origin_and_rotation(yaw):
    m=dict(x=-8.415,y=-44.866,resolution=.05,width=546,height=920,yaw=yaw)
    for p in [(0,0),(12,100),(545,919)]:
        world=pixel_to_world(m,*p)
        assert world_to_pixel(m,*world)==pytest.approx(p)


def test_manual_heading_is_in_world_not_screen():
    m=dict(x=1,y=2,resolution=.05,width=100,height=200,yaw=.3)
    x,y,a=manual_pose(m,(20,30),(40,30));assert a==pytest.approx(.3)
    assert (x,y)==pytest.approx(pixel_to_world(m,20,30))
    assert manual_pose(m,(20,30),(20,10))[2]==pytest.approx(.3+math.pi/2)


@pytest.mark.parametrize('start,end',[((-1,10),(5,10)),((100,10),(5,10)),((10,10),(10,10)),((float('nan'),10),(20,10))])
def test_bad_manual_point_rejected(start,end):
    with pytest.raises(ValueError):manual_pose(dict(x=0,y=0,resolution=.05,width=100,height=100,yaw=0),start,end)
