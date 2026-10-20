"""One-time user-requested correction; does not start ROS or move hardware."""
from pathlib import Path
from autolife_shelf_grasp_306.core import ProductStore,product_category

root=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306/data')
store=ProductStore(root)
p=next(v for v in store.products if v['id']=='017f1f54e7fc' and v['name']=='口香糖')
if (round(p['diameter_m'],4),round(p['height_m'],4)) not in ((.08,.05),(.05,.08)):
    raise RuntimeError('用户已另行修改尺寸，停止自动覆盖')
result=store.update(p['id'],name=p['name'],diameter_mm=50,height_mm=80,geometry='cylinder',category=product_category(p))
print('口香糖已保存：宽50mm、高80mm；样本保留',len(result['examples']))
