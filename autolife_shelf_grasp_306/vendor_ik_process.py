"""把原厂IK留在原厂Python环境，避免感知环境的Pinocchio/KDL二进制冲突。

协议只接受FK/IK数值计算，完全不导入或创建机器人控制器。
"""
import contextlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


class VendorIKProcess:
    def __init__(self, cfg, model, base, tip, locked, checkpoint=lambda:None):
        self.checkpoint=checkpoint;self._closed=False
        python = Path(cfg['vendor_python_path']).parents[2]/'bin/python'
        if not python.is_file():
            raise ValueError('原厂IK Python不存在：'+str(python))
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env['LD_LIBRARY_PATH'] = ':'.join(v for v in env.get('LD_LIBRARY_PATH','').split(':')
            if v and '/s3_perception/' not in v)
        self.process = subprocess.Popen([str(python), '-u', str(Path(__file__).resolve()),
            str(model), base, tip, json.dumps(locked)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env=env)
        try:
            ready = self.read(15.)
            if ready != {'ready': True}:
                raise ValueError('原厂IK启动协议错误')
        except Exception:
            self.close();raise

    def read(self, timeout):
        deadline=time.monotonic()+timeout
        while True:
            self.checkpoint()
            if select.select([self.process.stdout],[],[],min(.05,max(0.,deadline-time.monotonic())))[0]:break
            if time.monotonic()>=deadline:raise ValueError('原厂IK计算超时')
        line = self.process.stdout.readline()
        if not line:
            raise ValueError('原厂IK计算进程退出；未生成运动目标')
        response = json.loads(line)
        if 'error' in response:raise ValueError(response['error'])
        return response

    def request(self, kind, **values):
        self.process.stdin.write(json.dumps(dict(kind=kind, **values))+'\n')
        self.process.stdin.flush()
        return self.read(5.)['result']

    def close(self):
        if self._closed:return
        self._closed=True
        p = self.process
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=2.)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=2.)
        p.stdin.close();p.stdout.close()


def worker():
    # 原厂库导入会打印提示，全部导向stderr，stdout仅传输JSON。
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from autolife_robot_sdk.utils.kdl_ik_solver import KdlSolver
        import tempfile
        import xml.etree.ElementTree as ET
        from scipy.spatial.transform import Rotation
        import math
        model, base, tip, initial_locked = sys.argv[1:]
        allowed=set(json.loads(initial_locked))
        def build(locked):
            if set(locked)!=allowed or any(not math.isfinite(float(v)) or abs(float(v))>360 for v in locked.values()):
                raise ValueError('固定关节配置无效')
            tree = ET.parse(model)
            # 固定关节仍保留实测转角对应的完整安装变换，避免给IK多余自由度。
            for name, angle in locked.items():
                joint=tree.getroot().find(f"joint[@name='{name}']")
                origin=joint.find('origin')
                if origin is None:origin=ET.SubElement(joint,'origin',xyz='0 0 0',rpy='0 0 0')
                rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
                axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                rotation=Rotation.from_euler('xyz',rpy)*Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.deg2rad(angle))
                origin.set('rpy',' '.join(format(v,'.17g') for v in rotation.as_euler('xyz')))
                joint.set('type','fixed')
            with tempfile.TemporaryDirectory(prefix='vendor_ik_fixed_') as tmp:
                path=Path(tmp)/'fixed.urdf';tree.write(path)
                return KdlSolver(str(path), [base], [tip])
        solver=build(json.loads(initial_locked))
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request['kind']=='configure':
                    solver=build(request['locked']);result=[]
                elif request['kind']=='fk':
                    result = solver.fk([base],[tip],[np.asarray(request['q'])])[0]
                elif request['kind']=='ik':
                    seed=np.asarray(request['seed']);lower=np.asarray(request['lower']);upper=np.asarray(request['upper'])
                    solutions = solver.ik([base],[tip],[request['pose']],[seed],
                        ik_pos_solver_type='nr_jl',kwargs_ik_pos={
                            'q_min':lower,'q_max':upper,'maxiter':250,'eps':1e-5})
                    if solutions:
                        result=solutions[0]
                    else:
                        # 同一目标的原厂LMA数值解，不改变抓点或放宽限位。
                        solutions=solver.ik([base],[tip],[request['pose']],[seed],
                            ik_pos_solver_type='lma',kwargs_ik_pos={'maxiter':500,'eps':1e-5})
                        if not solutions:raise ValueError('原厂NR/LMA均未找到此位姿的解')
                        result=np.asarray(solutions[0])
                        result=seed+(result-seed+np.pi)%(2*np.pi)-np.pi
                        if np.any(result<lower-1e-6) or np.any(result>upper+1e-6):
                            raise ValueError('原厂LMA解超出腰臂限位')
                else:raise ValueError('仅支持无动作FK/IK')
            response = {'result':list(map(float,result))}
        except Exception as exc:
            response = {'error':str(exc)}
        print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':worker()
