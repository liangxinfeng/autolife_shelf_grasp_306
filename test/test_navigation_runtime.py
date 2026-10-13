import sys,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import autolife_shelf_grasp_306.navigation_runtime as module


def make_runtime(monkeypatch,tmp_path,conflicts=(),slam=(),owners=(),active=False):
    calls=[];events=[]
    fake=types.ModuleType('control_center_shell.preflight')
    fake.module_process_conflicts=lambda key:list(conflicts if key=='navigation' else slam)
    fake._processes=lambda:{}
    fake.record_process=lambda key,pid:calls.append(('record',pid))
    shell=types.ModuleType('control_center_shell');shell.preflight=fake
    monkeypatch.setitem(sys.modules,'control_center_shell',shell);monkeypatch.setitem(sys.modules,'control_center_shell.preflight',fake)
    def command(args,**kw):
        calls.append(tuple(args));return NS(returncode=0 if active or 'stop' in args else 3,stderr='')
    def spawn(args,**kw):
        calls.append(('spawn',args,kw));return NS(pid=98765,poll=lambda:None)
    monkeypatch.setattr(module.subprocess,'run',command);monkeypatch.setattr(module.subprocess,'Popen',spawn)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    runtime=module.NavigationRuntime(tmp_path,lambda event,**data:events.append((event,data)),lambda:list(owners))
    map_path=tmp_path/'map.yaml';map_path.write_text('image: map.png')
    return runtime,map_path,calls,events


def test_existing_stack_is_reused_without_service_mutation(monkeypatch,tmp_path):
    r,p,c,e=make_runtime(monkeypatch,tmp_path,conflicts=[{'pid':1}]);r._start(p)
    assert not c and not r.error and '不会重复启动' in e[-1][1]['text']


def test_slam_is_never_killed_to_make_room(monkeypatch,tmp_path):
    r,p,c,e=make_runtime(monkeypatch,tmp_path,slam=[{'pid':1}]);r._start(p)
    assert not c and '建图' in r.error


def test_foreign_map_blocks_launch(monkeypatch,tmp_path):
    r,p,c,e=make_runtime(monkeypatch,tmp_path,owners=['foreign_map']);r._start(p)
    assert '地图发布者' in r.error and not any(v[0]=='spawn' for v in c)


def test_launch_same_stack_and_never_restart_hardware(monkeypatch,tmp_path):
    r,p,c,e=make_runtime(monkeypatch,tmp_path,active=True);r._start(p)
    assert not r.error
    spawn=next(v for v in c if v[0]=='spawn')
    assert 'bringup_all.launch.py' in spawn[1] and 'map_yaml:='+str(p) in spawn[1]
    assert spawn[2]['start_new_session']
    stops=[v for v in c if v[0]=='systemctl' and 'stop' in v]
    assert stops==[('systemctl','--user','stop','gv-slam-service.service')]
    assert ('record',98765) in c


def test_failure_to_stop_vendor_prevents_start(monkeypatch,tmp_path):
    r,p,c,e=make_runtime(monkeypatch,tmp_path,active=True)
    monkeypatch.setattr(module.subprocess,'run',lambda args,**kw:NS(returncode=2 if 'stop' in args else 0,stderr='denied'))
    r._start(p);assert '未停止' in r.error and not any(v[0]=='spawn' for v in c)
