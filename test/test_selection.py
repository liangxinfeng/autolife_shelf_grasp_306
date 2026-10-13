import numpy as np
from autolife_shelf_grasp_306.selection import resolve_selected

def item(i=1,dx=0,product=None,feature=None):
    mask=np.zeros((100,160),bool);mask[20:80,30+dx:70+dx]=True
    return dict(id=i,mask=mask,feature=np.array([1.,0.]) if feature is None else feature,
                product=product,name='paper')

def test_id_renumbered_does_not_change_selected_object():
    chosen=item();fresh=item(13,dx=5)
    result,diag=resolve_selected(chosen,[fresh])
    assert result['id']==1 and diag['detector_id']==13
    np.testing.assert_array_equal(result['mask'],fresh['mask'])

def test_catalog_threshold_flicker_both_directions():
    product={'id':'water','diameter_m':.07}
    a,_=resolve_selected(item(product=product),[item(2)])
    assert a['product']==product
    b,_=resolve_selected(item(),[item(2,product=product)])
    assert b['product'] is None

def test_changed_catalog_identity_is_not_silently_accepted():
    result,_=resolve_selected(item(product={'id':'a'}),[item(2,product={'id':'b'})])
    assert result is None

def test_disappeared_moved_different_or_ambiguous_target_stops():
    for candidates in ([],[item(2,dx=65)],[item(2,feature=np.array([0.,1.]))],[item(2,dx=2),item(3,dx=-2)]):
        result,diag=resolve_selected(item(),candidates)
        assert result is None and diag['reason']

def test_low_appearance_similarity_not_rescued_by_same_id():
    assert resolve_selected(item(),[item(feature=np.array([.1,1.]))])[0] is None

def test_new_parameters_of_same_product_not_overwritten():
    result,_=resolve_selected(item(product={'id':'a','height_m':.05}),[item(product={'id':'a','height_m':.08})])
    assert result['product']['height_m']==.08
