"""整次规划共用的取消、时间预算和阶段提示；本模块不访问机器人。

一条用户规划请求只创建一个实例，左右臂和收回阶段共用；不能每次重试重建。
在候选循环、IK目标函数和路径逐点检查中调用 checkpoint，耗时操作前后都检查。
两个终止异常不继承 ValueError，防止被“候选不可用→试下一条”逻辑吞掉。
"""
import math
import time


class PlanningCancelled(RuntimeError):
    """用户取消整次请求，不能换臂或换候选继续运行。"""


class PlanningTimeout(RuntimeError):
    """整次请求预算耗尽，保留当前候选诊断后返回界面。"""


class PlanningControl:
    def __init__(self, stopped=lambda: False, on_status=lambda text: None,
                 timeout_sec=45.0, update_interval_sec=0.5, clock=time.monotonic):
        for name, value in [('规划时限', timeout_sec), ('提示间隔', update_interval_sec)]:
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + '必须为有限正数')
        self.stopped, self.on_status, self.clock = stopped, on_status, clock
        self.timeout_sec = float(timeout_sec)
        self.update_interval_sec = float(update_interval_sec)
        self.started = clock()
        self.deadline = self.started + self.timeout_sec
        self.arm = None
        self.phase = '准备规划'
        self.candidate = None
        self.total_candidates = None
        self.last_emitted_key = None
        self.last_emitted_time = -float('inf')
        self.terminal = None

    def _label(self, elapsed):
        prefix = {'left': '左臂', 'right': '右臂', None: '路径规划'}[self.arm]
        candidate = ''
        if self.candidate is not None:
            candidate = f' · 候选{self.candidate}'
            if self.total_candidates is not None:
                candidate += f'/{self.total_candidates}'
        return f'{prefix} · {self.phase}{candidate} · 已用{elapsed:.1f}秒'

    def checkpoint(self):
        """检查取消和全局预算。频繁调用时只按间隔发送文本，避免刷新拖慢计算。"""
        if self.terminal is not None:
            cls, message = self.terminal
            raise cls(message)
        if self.stopped():
            self.terminal = (PlanningCancelled, '用户已停止规划，未生成可执行路径')
            raise PlanningCancelled(self.terminal[1])
        now = self.clock()
        elapsed = max(0.0, now - self.started)
        if now >= self.deadline:
            message = self._label(elapsed) + f'；整次规划达到{self.timeout_sec:g}秒时限'
            self.terminal = (PlanningTimeout, message)
            raise PlanningTimeout(message)
        key = (self.arm, self.phase)
        if key != self.last_emitted_key or now - self.last_emitted_time >= self.update_interval_sec:
            self.on_status(self._label(elapsed))
            self.last_emitted_key = key
            self.last_emitted_time = now

    def set_stage(self, phase, *, arm=None, candidate=None, total_candidates=None):
        """更新阶段，不重置时限。典型阶段：预抓取、接近物体、抬离、持物收回。"""
        if arm not in (None, 'left', 'right'):
            raise ValueError('未知规划手臂')
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError('规划阶段不能为空')
        for value in (candidate, total_candidates):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError('候选序号必须为正整数')
        if total_candidates is not None and candidate is not None and candidate > total_candidates:
            raise ValueError('候选序号超出总数')
        self.arm, self.phase = arm, phase
        self.candidate, self.total_candidates = candidate, total_candidates
        self.checkpoint()
