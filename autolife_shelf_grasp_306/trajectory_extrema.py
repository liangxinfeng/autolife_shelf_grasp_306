"""低阶轨迹多项式的真实峰值：检查端点和全部实驻点，不重复保守降速。"""
import numpy as np


def peak_absolute(coefficients):
    c=np.asarray(coefficients,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all():
        raise ValueError('轨迹多项式系数无效')
    candidates=[0.,1.]
    derivative=np.polynomial.polynomial.polyder(c)
    try:
        roots=np.polynomial.polynomial.polyroots(derivative)
        if not np.isfinite(roots).all():raise ValueError('轨迹极值计算无效')
        for z in roots:
            if abs(z.imag)<1e-7 and -1e-10<=z.real<=1.+1e-10:
                candidates.append(float(np.clip(z.real,0.,1.)))
        peak=max(abs(np.polynomial.polynomial.polyval(x,c)) for x in candidates)
        # 浮点舍入余量远小于编码器分辨率，不改变运动限值。
        return peak*(1.+1e-9)+np.finfo(float).eps*sum(abs(c))*32
    except (ValueError,np.linalg.LinAlgError,FloatingPointError):
        # 数值求根失败时保留原来的保守上界，绝不把失败当作无需限速。
        grid=np.linspace(0.,1.,129)
        lip=sum(i*abs(c[i]) for i in range(1,len(c)))
        return float(max(abs(np.polynomial.polynomial.polyval(grid,c)))+lip/256.)
