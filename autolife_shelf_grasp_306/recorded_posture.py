"""Read-only UI summary; never approximate or rewrite recorded joint targets."""
def summary(record):
    from control_center_joint_control.body_height import estimate_body_lowering
    try:
        frames=record['keyframes']
        start,residual=estimate_body_lowering(frames[0]['joints_deg'],signed=True)
        end,_=estimate_body_lowering(frames[-1]['joints_deg'],signed=True)
    except (KeyError,TypeError,ValueError,IndexError):
        return '请选择完整的示教动作；执行前校验全身关节记录',0.
    label=lambda x:('反向' if x<-.002 else '正向')+f'约{abs(x)*100:.1f}cm'
    text=f'起始{label(start)}，结束{label(end)}。按动作记录恢复踝、膝、腰及全过程高度变化；不是固定一个高度。'
    if residual>2.:text+='该姿态含独立腰部前倾，厘米仅供参考，以录制关节角为准。'
    return text,abs(start)*100.
