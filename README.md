# 云蝶300 · 视觉抓取与自动任务

本分支 `robot300-20261020` 是300机器人（192.168.8.122）当前 `~/ros2_ws/src/autolife_shelf_grasp_306` 功能包的源码快照（2026-10-20）。ROS包名保持不变；本次只创建新分支。

## 当前功能

- Grounding DINO + SAM 2 语义检测与分割、DINOv2 商品匹配、RGB-D几何定位。
- 抓取与放置规划、左右臂选择与独立标定配置、可设置抓取中心补偿及收紧量。
- 导航地图与航点、全局/手动重定位，以及导航、抓取、放置、固定动作的任务编排。
- 全身关键帧记录、编辑、预览与动作串回放，身体高度及头部姿态设置。
- 自动任务单次/循环运行、暂停继续、状态展示与全屏云蝶科技/UNDY页面。

功能说明见 [WORKFLOWS_ZH.txt](WORKFLOWS_ZH.txt)。其他历史说明见 [README_ZH.md](README_ZH.md)；历史文档中的旧识别后端和参数不代表当前配置，现用配置以 [config/grasp.yaml](config/grasp.yaml) 及源码为准。

## 环境与运行

面向Ubuntu 24.04、ROS 2 Jazzy及相同厂商基础环境，不是独立安装即可控制任意机器人的程序。依赖控制中心、手眼标定、遥操运动学/碰撞模块、厂商SDK与URDF、RGB-D相机接口和支持SYNC的机械臂服务。系统Python需要ROS/Qt依赖；`robot_env` 需要模型推理、运动学及碰撞计算依赖。

```bash
cd ~/ros2_ws/src
git clone -b robot300-20261020 https://github.com/liangxinfeng/autolife_shelf_grasp_306.git
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select autolife_shelf_grasp_306 --symlink-install
source install/setup.bash
ros2 run autolife_shelf_grasp_306 shelf_grasp
```

本分支保留300实机配置：`robot_id: 300`，复位采用系统位姿（`reset_pose_source: robot_system`），允许缺失标定时采用系统几何关系（`allow_system_geometry_fallback: true`）。左臂标定路径指向 `sessions/300_current/session.json`，右臂独立标定路径为空。模型、标定文件与机器人运行环境需另行部署；迁移至其他机型前应核对关节、夹爪、标定、位姿和路径配置。

## 发布范围

包含当前源码、配置、启动脚本、测试及文档；不包含 `models/` 权重、`data/` 商品照片/特征库/动作记录/任务运行数据、标定会话、缓存、日志或旧代码备份。缺少这些外部依赖时，单独克隆仓库并不等于完整部署。

本次发布保持机器人源码和配置原样，仅更新本README的机型、分支及部署说明；进行了源文件哈希核对与Python语法检查；不代表全部功能重新完成实机测试。测试目录中包含依赖ROS、模型和现场数据的脚本，不应在机器人上未经确认批量执行。软件停止不能代替实体急停；运动前需核对现场条件和控制权。

公开源码不等于授予第三方商业使用许可；各模型、开源库及厂商组件遵循各自授权。
