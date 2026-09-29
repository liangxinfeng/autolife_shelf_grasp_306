"""Small pure functions: latest-only data, product storage and guarded geometry."""
import json
import os
import uuid
import copy
from pathlib import Path
import cv2
import numpy as np


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def unit(x):
    x = np.asarray(x, float)
    norm = np.linalg.norm(x)
    if not np.isfinite(x).all() or norm < 1e-9: raise ValueError('无效向量')
    return x/norm


def corrected_target(center,offset_mm):
    """Fixed torso-frame target correction, independent of camera/tool rotation.

    On this URDF, the right shoulder has positive torso Y (left has negative
    Y). Do not assume this model uses the ROS base-link left-positive axis.
    """
    offset=np.asarray(offset_mm,float);center=np.asarray(center,float)
    if (offset.shape!=(3,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>50.
            or center.shape!=(3,) or not np.isfinite(center).all()):
        raise ValueError('抓取目标补偿无效（最大50mm）')
    delta=offset/1000.
    return center+delta,{'observed_center_torso_m':center.tolist(),
        'target_offset_frame':'Link_Waist_Yaw_to_Shoulder_Inner',
        'target_offset_torso_mm':offset.tolist(),'target_offset_torso_m':delta.tolist()}


def match(feature, products, threshold=.72, margin=.06):
    """Cosine similarity is not confidence probability. Compare distinct SKUs."""
    query = unit(feature); scores=[]
    for product in products:
        refs = np.asarray(product.get('features', []), float)
        if refs.size == 0: continue
        if refs.ndim != 2 or refs.shape[1] != len(query) or not np.isfinite(refs).all():
            raise ValueError('商品特征格式不兼容，请重新录入')
        norms=np.linalg.norm(refs,axis=1)
        if np.any(norms<1e-9): raise ValueError('商品特征损坏')
        values = (refs/norms[:,None])@query
        # Strongest two examples reduce a single accidental reference match.
        scores.append((float(np.mean(np.sort(values)[-2:])), product['id']))
    scores.sort(reverse=True)
    if not scores: return {'id':None, 'score':0., 'margin':0.}
    gap=scores[0][0]-(scores[1][0] if len(scores)>1 else -1.)
    return {'id':scores[0][1] if scores[0][0]>=threshold and gap>=margin else None,
            'score':scores[0][0], 'margin':gap}


class ProductStore:
    def __init__(self, root):
        self.root=Path(root); self.path=self.root/'products.json'
        self.reload()

    def reload(self):
        data=json.loads(self.path.read_text('utf-8')) if self.path.exists() else {'version':1,'products':[]}
        if data.get('version')!=1: raise ValueError('商品库版本不兼容')
        self.products=data['products']

    @staticmethod
    def parameters(name,diameter_mm,height_mm,geometry='cylinder',depth_mm=None,category='general'):
        name=name.strip()
        if not 1<=len(name)<=60: raise ValueError('请输入1～60字的商品名称')
        if geometry not in ('cylinder','solid','bag_edge'):raise ValueError('物品类型无效')
        if category not in ('beverage','general'):raise ValueError('商品分类无效')
        if not 15<=diameter_mm<=(350 if geometry=='bag_edge' else 90) or not 40<=height_mm<=400:
            raise ValueError('实体夹取宽度15～90mm；袋子整体宽度15～350mm；高度40～400mm')
        depth_mm=diameter_mm if geometry=='cylinder' else float(depth_mm or 0)
        if not np.isfinite(depth_mm) or not (1 if geometry=='bag_edge' else 5)<=depth_mm<=(20 if geometry=='bag_edge' else 150):
            raise ValueError('实体厚度5～150mm；袋口局部厚度1～20mm')
        return dict(name=name,diameter_m=diameter_mm/1000.,height_m=height_mm/1000.,
                    geometry=geometry,depth_m=depth_mm/1000.,category=category)

    def save(self,products):
        # Keep previous library revisions and removed sample images recoverable.
        if self.path.exists():
            atomic_json(self.root/'history'/(uuid.uuid4().hex+'.json'),json.loads(self.path.read_text('utf-8')))
        atomic_json(self.path,{'version':1,'feature_model':'dinov2_vits14','products':products})
        self.products=products

    def add(self, name, diameter_mm, height_mm, feature, image,geometry='cylinder',depth_mm=None,category='general'):
        params=self.parameters(name,diameter_mm,height_mm,geometry,depth_mm,category)
        vector=unit(feature).tolist();self.reload();products=copy.deepcopy(self.products)
        p=next((p for p in products if p['name']==params['name']),None)
        if p is None:
            p={'id':uuid.uuid4().hex[:12],'features':[],'examples':[]};products.append(p)
        p.update(params)
        # Filenames are generated IDs, never user-provided text.
        image_path=self.root/'examples'/p['id']/(uuid.uuid4().hex+'.jpg')
        image_path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(image_path),image):raise ValueError('样本图像保存失败')
        p['features'].append(vector);p['examples'].append(str(image_path.relative_to(self.root)))
        self.save(products)
        return p

    def update(self,product_id,**values):
        params=self.parameters(**values);self.reload();products=copy.deepcopy(self.products)
        p=next((p for p in products if p['id']==product_id),None)
        if p is None:raise ValueError('商品已不存在，请刷新商品库')
        if any(v['id']!=product_id and v['name']==params['name'] for v in products):raise ValueError('名称已被其他商品使用')
        p.update(params);self.save(products);return p

    def remove_example(self,product_id,index):
        self.reload();products=copy.deepcopy(self.products)
        p=next((p for p in products if p['id']==product_id),None)
        if p is None or not isinstance(index,int) or not 0<=index<len(p['examples']):raise ValueError('样本不存在')
        if len(p['examples'])!=len(p['features']):raise ValueError('样本图片与特征数量不一致')
        p['examples'].pop(index);p['features'].pop(index);self.save(products)
        return p

    def remove_product(self,product_id):
        self.reload()
        removed=next((p for p in self.products if p['id']==product_id),None)
        if removed is None:raise ValueError('商品已不存在，请刷新商品库')
        # save() archives the full previous catalog; sample files stay available
        # for recovery but are no longer referenced or matched in either mode.
        self.save([copy.deepcopy(p) for p in self.products if p['id']!=product_id])
        return removed


def product_category(product):
    # Original bottle-only records predate geometry/category fields.
    return product.get('category','beverage' if 'geometry' not in product else 'general')


def crop_object(image, mask):
    ys,xs=np.where(mask)
    if len(xs)<80: raise ValueError('分割区域过小')
    x0,x1=max(0,xs.min()-4),min(image.shape[1],xs.max()+5)
    y0,y1=max(0,ys.min()-4),min(image.shape[0],ys.max()+5)
    crop=image[y0:y1,x0:x1].copy(); crop[~mask[y0:y1,x0:x1]]=127
    return crop


def depth_cloud(frame, mask, config, step=2):
    """Same-frame aligned depth only; no fill of unknown/transparent surfaces."""
    from autolife_handeye_calibration_307.depth_target import aligned_depth
    if frame.get('depth') is None: raise ValueError(frame.get('depth_error','等待深度'))
    aligned_depth(frame['image'], frame['depth'],frame['frame_id'],frame['depth_id'],
                  frame['intrinsics'],frame['depth_intrinsics'])
    if mask.shape!=frame['depth'].shape: raise ValueError('分割与深度尺寸不同')
    if np.any(np.abs(frame['D'])>1e-9) and frame['intrinsics']['model_id'] not in (0,4):
        raise ValueError('当前畸变模型未支持')
    safe=cv2.erode(mask.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
    ys,xs=np.where(safe); ys,xs=ys[::step],xs[::step]
    z=frame['depth'][ys,xs].astype(float)*config['depth_scale_m']
    lo,hi=config['depth_range_m']; good=(z>=lo)&(z<=hi)
    if len(z)<30 or good.mean()<.55: raise ValueError('有效深度不足；请避开反光/透明区域')
    xs,ys,z=xs[good],ys[good],z[good]
    median=np.median(z); keep=np.abs(z-median)<.065
    if keep.mean()<.65: raise ValueError('物体与背景深度混杂')
    xs,ys,z=xs[keep],ys[keep],z[keep]
    rays=cv2.undistortPoints(np.stack([xs,ys],axis=1).astype(float)[:,None,:],frame['K'],frame['D']).reshape(-1,2)
    return np.column_stack([rays*z[:,None],z])


def stages(center, yaw, config):
    from .bottle import horizontal_basis
    n,_,up=horizontal_basis(config['up_torso'],yaw); p=np.asarray(center,float)
    lifted=p+up*config['lift_m']
    return [('预抓取',p-config['pregrasp_m']*n),('抓取',p.copy()),
            ('抬起',lifted),('收回',lifted-config['withdraw_m']*n)]


def execution_blockers(cfg):
    if cfg.get('supervised_experiment') is True:
        return [] if cfg.get('hardware_execution') is True else ['未开放真机执行']
    labels={'hardware_execution':'未开放真机执行','camera_tcp_validated':'相机/TCP待实测验证',
            'gripper_mapping_validated':'夹爪开度映射待验证','tool_orientation_validated':'水平夹取方向待验证',
            'external_scene_validated':'桌面及周边空间待验证'}
    return [label for key,label in labels.items() if cfg.get(key) is not True]


def reset_blockers(cfg):
    # Reset has no visual target and must not depend on grasp validation flags.
    return [] if cfg.get('reset_hardware_enabled') is True else ['未开放复位运动']


def open_tolerance(cfg):
    value=float(cfg.get('gripper_open_tolerance_deg',4.))
    if not np.isfinite(value) or not 1.<=value<=5.:raise ValueError('全开容差配置须在1～5°以内')
    return value


def check_plan_identity(plan,run_id,plan_id,product_id):
    if (not plan_id or not product_id or plan.get('plan_id')!=plan_id or
        plan.get('run_id')!=run_id or plan.get('product',{}).get('id')!=product_id):
        raise ValueError('规划与当前窗口/选中饮料不一致，请重新规划')
