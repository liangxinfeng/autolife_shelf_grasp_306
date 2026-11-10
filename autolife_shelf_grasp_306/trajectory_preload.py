"""把原厂纯轨迹函数的首次导入与只读反馈准备并行；使能前必须完成。"""
import time
from concurrent.futures import ThreadPoolExecutor


class TrajectoryPreload:
    def __init__(self,loader,cfg):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='grasp_import')
        self.future=self.pool.submit(loader,cfg)

    def result(self,poll,timeout=10.):
        deadline=time.monotonic()+timeout
        while not self.future.done():
            poll()  # 主线程仍响应取消、界面心跳和反馈；此时尚未使能。
            if time.monotonic()>deadline:raise ValueError('原厂轨迹函数加载超时，尚未使能')
        return self.future.result()  # 导入失败必须在使能之前传播。

    def close(self):
        self.pool.shutdown(wait=False,cancel_futures=True)
