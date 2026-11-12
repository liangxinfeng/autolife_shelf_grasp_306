"""官方URDF的只读坐标转换；仅计算几何，不实现运动执行或逆解。"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
AppError = ValueError


class ModelGeometry:
    def __init__(self,official_urdf):
        path=Path(official_urdf)
        # 完整模型与官方运行模型必须同目录、同名称族，不能跨型号寻找一个能用的模型。
        candidates=[path,path.with_name(path.name.replace('_simplified',''))]
        self.path=None
        for candidate in dict.fromkeys(candidates):
            if not candidate.is_file():continue
            tree=ET.parse(candidate).getroot();links={v.get('name') for v in tree.findall('link')}
            if 'Link_Camera_Head_Forehead' in links:self.path=candidate;break
        if self.path is None:raise AppError('本机官方完整URDF缺少头部相机安装连杆，请检查部署的模型文件')
        self.joints=[]
        for j in tree.findall('joint'):
            origin=j.find('origin');axis=j.find('axis');mimic=j.find('mimic')
            xyz=[float(v) for v in (origin.get('xyz','0 0 0') if origin is not None else '0 0 0').split()]
            rpy=[float(v) for v in (origin.get('rpy','0 0 0') if origin is not None else '0 0 0').split()]
            t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();t[:3,3]=xyz
            self.joints.append(dict(name=j.get('name'),kind=j.get('type'),parent=j.find('parent').get('link'),child=j.find('child').get('link'),origin=t,
                axis=np.array([float(v) for v in (axis.get('xyz','1 0 0') if axis is not None else '1 0 0').split()]),mimic=mimic.attrib if mimic is not None else None))
        roots=links-{j['child'] for j in self.joints}
        if len(roots)!=1:raise AppError('官方URDF根坐标不唯一')
        self.root=roots.pop()
        # URDF拓扑及旋转轴只解析一次；每帧仍用所有当前关节值计算。
        ordered=[];available={self.root};todo=list(self.joints)
        while todo:
            ready=[j for j in todo if j['parent'] in available]
            if not ready:raise AppError('官方URDF连杆树不完整')
            for j in ready:
                ordered.append(j);available.add(j['child']);todo.remove(j)
        self._ordered=ordered
        self._rotating=[j for j in ordered if j['kind'] in ('revolute','continuous') and 'Gripper' not in j['name']]
        self._axes=np.array([j['axis']/np.linalg.norm(j['axis']) for j in self._rotating])
        self._origins=np.array([j['origin'][:3,:3] for j in self._rotating])
        for i,j in enumerate(self._rotating):j['rotation_index']=i

    def frames(self,values):
        angles=[]
        for j in self._rotating:
            m=j['mimic']
            angles.append(math.radians(values.get(m['joint'],0))*float(m.get('multiplier',1))+float(m.get('offset',0))
                          if m else math.radians(values.get(j['name'],0)))
        rotations=(self._origins@Rotation.from_rotvec(self._axes*np.asarray(angles)[:,None]).as_matrix()
                   if angles else [])
        frames={self.root:np.eye(4)}
        for j in self._ordered:
            t=j['origin'].copy()
            if 'rotation_index' in j:t[:3,:3]=rotations[j['rotation_index']]
            elif j['kind']=='prismatic':t[:3,3]+=j['axis']*values.get(j['name'],0.)
            frames[j['child']]=frames[j['parent']]@t
        return frames

    def tool_transform(self,side="right"):
        f=self.frames({});prefix='Link_'+side.title();wrist=f[prefix+'_Wrist_Lower_to_Gripper']
        tcp=np.mean([(np.linalg.inv(wrist)@f[prefix+'_Gripper_'+s+'_Finger'])[:3,3] for s in ('Left','Right')],axis=0)
        t=np.eye(4);t[:3,3]=tcp
        # 工具X沿夹爪安装面→两指中心，Y使用夹爪安装系的横向，Z由右手系确定。
        # S3腕系的夹爪向前轴接近-Z，不能沿用旧Rx(-90)把腕+X当成接近轴。
        inv=np.linalg.inv(wrist)
        mount=(inv@f[prefix+'_Gripper'])[:3,3]
        left=(inv@f[prefix+'_Gripper_Left_Finger'])[:3,3]
        right=(inv@f[prefix+'_Gripper_Right_Finger'])[:3,3]
        forward=tcp-mount
        length=float(np.linalg.norm(forward))
        if not np.isfinite(length) or length<1e-6:
            raise AppError('URDF夹爪安装面与两指中心重合，不能确定接近轴')
        forward=forward/length
        # 手指连杆原点是偏置的转轴安装点，不是指腹中心；直接连线会
        # 把约17度机构偏置误当成夹爪的横滚，使“水平抓取”实际倾斜。
        # 指腹横向沿安装坐标Y；连线仅用于保持左右符号一致。
        closing=(inv@f[prefix+'_Gripper'])[:3,1].copy()
        if closing@(left-right)<0:closing=-closing
        closing=closing-forward*float(closing@forward)
        span=float(np.linalg.norm(closing))
        if not np.isfinite(span) or span<1e-6:
            raise AppError('URDF两指方向退化，不能确定夹爪开合轴')
        closing=closing/span
        t[:3,:3]=np.column_stack((forward,closing,np.cross(forward,closing)))
        return t

