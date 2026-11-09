"""任务抓取与单次抓取使用相同起点模式；不发运动指令。"""
import yaml


def requires_reset(root):
    path=root/'config/grasp.yaml'
    if not path.exists():return True
    return bool(yaml.safe_load(path.read_text('utf-8')).get('require_reset_before_grasp',True))
