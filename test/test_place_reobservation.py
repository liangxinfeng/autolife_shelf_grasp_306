import numpy as np
from autolife_shelf_grasp_306.demo_surface import obstacle_boxes,retryable_placement_error


TABLE=dict(coef=[0,0,.75],minimum=[.2,-.3,.75],maximum=[.9,.3,.75],
           support_method='observed_semantic_plane')
ORIGIN=np.array([0.,0.,1.3])


def patch(center,radius=.001):
    x,y=np.meshgrid(np.linspace(-radius,radius,7),np.linspace(-radius,radius,7))
    return np.array(center)+np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]


def behind(points):
    rays=points-ORIGIN
    return ORIGIN+rays*((.75-ORIGIN[2])/rays[:,2])[:,None]


def test_positive_later_ray_clears_old_tall_depth_ghost():
    old=patch([.5,0,.85]);latest=behind(old)
    assert obstacle_boxes(old,TABLE)
    assert not obstacle_boxes(latest,TABLE,frames=[old,latest],camera_origin=ORIGIN)


def test_latest_object_is_never_cleared_by_old_empty_surface():
    newest=patch([.5,0,.85]);old=behind(newest)
    assert obstacle_boxes(newest,TABLE,frames=[old,newest],camera_origin=ORIGIN)


def test_persistent_thin_object_remains():
    obj=patch([.5,0,.78])
    assert obstacle_boxes(obj,TABLE,frames=[obj]*5,camera_origin=ORIGIN)


def test_unknown_or_occluded_rays_do_not_clear_old_objects():
    old=patch([.5,0,.85])
    for newest in (patch([.5,.15,.75]),ORIGIN+(old-ORIGIN)*.8):
        boxes=obstacle_boxes(newest,TABLE,frames=[old,newest],camera_origin=ORIGIN)
        assert any(b['maximum'][0]>.49 and b['minimum'][0]<.51 for b in boxes)


def test_same_xy_but_different_ray_does_not_clear_overhang():
    old=patch([.5,0,.85]);below=patch([.5,0,.75])
    assert obstacle_boxes(below,TABLE,frames=[old,below],camera_origin=ORIGIN)


def test_retry_is_bounded_and_only_for_clutter_rejections():
    diag=[dict(error='放置轨迹接近桌面已有物品#5')]
    assert retryable_placement_error(diag,0)
    assert retryable_placement_error(diag,1)
    assert not retryable_placement_error(diag,2)
    assert not retryable_placement_error([dict(error='机器人自碰撞')],0)
    assert not retryable_placement_error([dict(error='位置误差8mm')],0)
