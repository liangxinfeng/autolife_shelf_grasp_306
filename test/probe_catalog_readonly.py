"""Real worker/camera, isolated COPY of catalog; never prepare or execute."""
import hashlib,json,os,queue,shutil,subprocess,tempfile,threading,time
from pathlib import Path

source=Path('/home/ubuntu/ros2_ws/src/autolife_shelf_grasp_306')
before=hashlib.sha256((source/'data/products.json').read_bytes()).hexdigest()
root=Path(tempfile.mkdtemp(prefix='shelf_catalog_probe_'));(root/'data').mkdir();(root/'run').mkdir()
shutil.copytree(source/'config',root/'config');(root/'models').symlink_to(source/'models',target_is_directory=True)
shutil.copy2(source/'data/products.json',root/'data/products.json')
shutil.copytree(source/'data/examples',root/'data/examples')
os.environ['ROS_DOMAIN_ID']='0';os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
messages=queue.Queue()
with (root/'worker.log').open('w') as log:
    proc=subprocess.Popen(['/home/ubuntu/miniconda3/envs/robot_env/bin/python','-m','autolife_shelf_grasp_306.worker',
        '--root',str(root),'--run',str(root/'run')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
    def reader():
        for line in proc.stdout:
            if line.startswith('{'):messages.put(json.loads(line))
    threading.Thread(target=reader,daemon=True).start()
    def wait_event(kind):
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            event=messages.get(timeout=max(.01,deadline-time.monotonic()))
            if event['event'] in ('error','fatal'):raise AssertionError(event)
            if event['event']==kind:return event
        raise TimeoutError(kind)
    ident=0
    def request(action,event,**values):
        global ident
        ident+=1;proc.stdin.write(json.dumps(dict(action=action,request_id=ident,**values),ensure_ascii=False)+'\n');proc.stdin.flush()
        result=wait_event(event);assert result['request_id']==ident;return result
    try:
        wait_event('ready')
        cat=request('catalog','catalog');gum=next(p for p in cat['products'] if p['name']=='口香糖');count=len(gum['examples'])
        values=dict(name='口香糖',diameter_mm=50,height_mm=80,geometry='cylinder',category='general')
        cat=request('update_product','catalog',product_id=gum['id'],values=values)
        p=next(p for p in cat['products'] if p['id']==gum['id']);assert p['height_m']==.08 and len(p['examples'])==count
        scene=request('set_mode','scene',mode='all');assert scene['mode']=='all'
        selected=next((v for v in scene['items'] if v['product_id']==gum['id']),None)
        assert selected,('gum not matched',scene['items'])
        assert selected['dimensions_mm'][:2]==[50.,80.]
        updated=request('enroll','enrolled',id=selected['id'],**values)
        cat=request('catalog','catalog');p=next(p for p in cat['products'] if p['id']==gum['id']);assert len(p['examples'])==count+1
        cat=request('remove_sample','catalog',product_id=gum['id'],index=count)
        assert len(next(p for p in cat['products'] if p['id']==gum['id'])['examples'])==count
        drinks=request('set_mode','scene',mode='beverages');assert drinks['mode']=='beverages'
        for item in drinks['items']:
            if item.get('product_id')==gum['id']:assert item['dimensions_mm'][:2]==[50.,80.]
        assert json.loads((root/'data/settings.json').read_text())['recognition_mode']=='beverages'
        print('CATALOG_WORKER_OK',{'all_candidates':len(scene['items']),'drink_candidates':len(drinks['items']),
            'dimensions_mm':selected['dimensions_mm'],'isolated_root':str(root)},flush=True)
    finally:
        proc.stdin.write('{"action":"quit"}\n');proc.stdin.flush()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
assert hashlib.sha256((source/'data/products.json').read_bytes()).hexdigest()==before
