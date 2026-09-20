# 云蝶306 · 视觉抓取实验

306机器人 `~/grasp306` 对应功能包，提供物品识别、商品录入、深度定位、抓取规划和执行界面。源码取自306当前功能包（2026-09-20导出）。

## 版本区分

- **`robot306-20260920`**：本分支，306当前源码、配置与使用说明。
- **`main`**：保留此前上传的2026-09-16版本，不覆盖、不替换。
- 本次仅归档源码，不修改机器人运行程序，不启动复位、抓取或导航。

## 功能

- **仅饮料 / 通用物品**两种模式，共用商品库，支持修改尺寸、增删样本图片。
- YOLO11分割与FastSAM通用候选 → DINOv2商品匹配 → 多帧RGB-D定位。
- 使用手眼标定外参和夹爪TCP，规划预抓取、抓取、抬起、收回完整流程。
- 优先正向接近，必要时逐步尝试小角度；收回方向单独求解，避免仅为收回困难而增大抓取角度。
- 一键复位并张开夹爪；确认抓取后自动闭爪、抬起和收回，保留停止按钮。
- 抓取窗口中可手动控制底盘，调整机器人与货架的相对位置。
- 对接控制中心导航，显示小地图、切换地图、查看及选择共享航点。
- 演示流程：前台 → 货架 → 识别并抓取 → 固定持物姿态收回 → 导航返回前台 → 识别桌面并放置饮料。
- 支持部分可见物体的定位；录入高度仅作参考，规划仍检查实际深度、运动学及执行条件。

## 环境与安装

面向Ubuntu 24.04、ROS 2 Jazzy和306厂商运行环境，**不是通用机器人即装即用软件**。需要：

- [手眼标定功能包](https://github.com/liangxinfeng/autolife_handeye_calibration_306)。
- [306控制中心分支](https://github.com/liangxinfeng/autolife_robot_control_center/tree/robot306-20260920)中的关节、底盘控制权与导航模块。
- [遥操V4](https://github.com/liangxinfeng/openarmx_teleop_vr_306_v4)的运动学与碰撞模块。
- 厂商SDK、机器人URDF、相机共享内存及支持保持式SYNC入口的机械臂服务；这些不随仓库分发。
- 系统Python的ROS/Qt依赖，以及 `robot_env` 内的PyTorch、Ultralytics、Pinocchio、碰撞库等依赖。

```bash
cd ~/ros2_ws/src
git clone --branch robot306-20260920 https://github.com/liangxinfeng/autolife_shelf_grasp_306.git
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select autolife_shelf_grasp_306 --symlink-install
source install/setup.bash
ros2 launch autolife_shelf_grasp_306 grasp.launch.py
```

已有306环境直接运行 `~/grasp306`。首次安装可在检查路径后，将 `scripts/grasp306` 复制到用户主目录并添加执行权限。

## 使用前配置

编辑 [config/grasp.yaml](config/grasp.yaml)：相机/模型路径、标定会话、机器人模型、TCP与目标补偿、运动参数。

- 模型权重及DINOv2源码按 [详细说明](README_ZH.md) 单独部署，本仓库不包含权重。
- 配置保留306实机参数；其中标定会话绝对路径在新机器上不存在，必须改成该机器自己的结果。
- `tcp_offset_mm: 15` 是靶标到夹取中心的工具偏移；`target_offset_torso_mm: [0,15,0]` 是另一个向机器人右侧15mm的经验补偿，不能混为一谈，也不能直接套用其他机器人。
- 新环境先关闭 `hardware_execution`、`reset_hardware_enabled` 和 `supervised_experiment`，完成实机核验后再按现场流程开放运动。仓库配置是306现用快照，不是安全默认模板。

## 操作

启动 → 复位并准备 → 识别/录入物品 → 选择物品 → 规划预览 → 确认抓取。

复位、确认抓取均会驱动真机；仅启动界面不会自动运动。不要同时运行其他机械臂控制程序。

演示模式需先在控制中心启动导航并完成定位，再选择前台、货架航点及商品。放置参数可在界面调整。详见[航点演示](NAVIGATION_DEMO.md)、[底盘控制](CHASSIS_ZH.md)和[部分可见目标](VISIBLE_TARGET_ZH.md)。此流程依赖相机视野、机器人可达空间和现场安全条件，不保证所有场景一次成功。

## 边界与数据

当前属于**有人看护的实验系统**，不保证抓取成功或标定精度。保留关节限制、自碰撞、反馈时效、控制权和轨迹检查，但没有完整的货架、桌面和邻物外部碰撞感知。软件停止不能替代实体急停；请清空运动空间并防止物品跌落。

未上传商品照片/特征库、标定会话、运行日志、模型权重、厂商二进制及现场图像；程序运行后在本地生成数据。纯逻辑测试位于 `test/`，部分回放测试需要另备现场数据，连接硬件的诊断脚本不应批量执行。

详细使用与版本记录：[README_ZH.md](README_ZH.md)、[VALIDATION_ZH.md](VALIDATION_ZH.md)。公开源码不等于授予第三方商业使用许可；保留现有许可声明，YOLO/FastSAM、DINOv2及厂商组件分别遵循其原有授权。
