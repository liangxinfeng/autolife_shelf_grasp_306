# 与2026-11-09抓取版本配套的原厂扩展

这些文件记录本次抓取修改所依赖的**增量**，不包含厂商完整SDK，也不提供自动重启或自动运动脚本。

| 文件 | 用途 |
|---|---|
| `main.py.patch` | 在既有V5/SYNC原厂入口基础上加入抓取结束保持和带物SYNC交接，修复收回结束/放置开始松爪 |
| `process_worker.py.patch` | 在已有原厂电机工作进程内挂接只读反馈刷新服务 |
| `grasp_feedback_refresh.py` | 只读反馈刷新扩展；复用原厂总线与接收线程，不创建新的硬件控制器 |
| `versions.json` | 每个补丁的基线SHA-256及345已安装结果SHA-256 |

补丁目标目录是当前 `robot_env` 中的 `autolife_robot_arm` Python包。`main.py.patch`的基线是345本机2026-11-09抓取结束保持修正前的入口，**已经带有此前V5/SYNC功能**，不是任意原厂发行版。`process_worker.py.patch`基线为同机接入只读刷新前版本。

补丁采用Linux LF换行。`base_sha256`对应统一为LF后的基线，`archived_base_sha256_before_lf_normalization`保留本地历史备份原始哈希。应用前先核对基线文件内容/换行及 `git apply --check`，有代码差异应对照接口人工合并；不能强行覆盖不同机器人或不同SDK的入口。本次仅存档补丁，没有将这些文件自动部署到其他机器。

已安装结果可用 `versions.json` 逐文件核对。修改原厂入口需要在现场停止关节任务、做好支撑后重新加载原厂服务；直接构建本ROS包不会替你完成该操作。

本机已验证的新接口、运行环境及未包含资源见 [部署依赖说明](../../docs/DEPLOYMENT_345.md)。
