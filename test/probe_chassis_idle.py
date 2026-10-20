"""Start helper with NO driving input; verify feedback and unchanged lease."""
import json,subprocess,time,redis
client=redis.Redis(host='127.0.0.1')
keys=['shared_params:0:300:is_navigating','shared_params:0:300:chassis_owner']
before=client.mget(keys)
p=subprocess.Popen(['/usr/bin/python3','-m','autolife_shelf_grasp_306.chassis'],
                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
try:
    time.sleep(2)
    output,err=p.communicate('',timeout=4)
    messages=[json.loads(line) for line in output.splitlines() if line.startswith('{')]
    assert p.returncode==0,(output,err)
    assert messages and all(not m['active'] for m in messages),messages
    assert any(m['stationary'] for m in messages),messages
    assert before==client.mget(keys),'外部控制状态改变，请检查，未发送运动输入'
    print('IDLE_FEEDBACK_OK',messages[-1])
finally:
    if p.poll() is None:p.terminate();p.wait(timeout=3)
    client.close()
