# 345源码快照部署依赖

## 当前机器实际环境

- 源码：`~/ros2_ws/src/autolife_shelf_grasp_306`，通过colcon symlink安装。
- UI/ROS：`/usr/bin/python3`、ROS 2 Jazzy、Qt5；系统Python不承担Pinocchio运算。
- 感知/规划：`~/ros2_ws/venvs/s3_perception/bin/python`。包括深度处理、Grounding DINO/SAM 2、DINOv2、Pinocchio及本机已有运动学/碰撞依赖。
- 原厂控制：`~/miniconda3/envs/robot_env/bin/python`，已有 `autolife_robot_arm` / `autolife_robot_sdk`，用户服务 `arm-control-service.service`。
- 当前桌面实际入口：`~/ros2_ws/bin/grasp345`，加载 `environment345.sh` 后启动本包；这两个属于机器部署环境，并非本源码包的标准ROS入口。
- 原有控制中心关节/底盘/导航模块、遥操导航/运动学模块、相机和模型网格依赖应保留。完整声明参考 `package.xml`，它不是pip可完全解决的独立依赖清单。

## 本次必须配套的原厂服务接口

当前节点后缀为 `0_345`，以实际原厂 `TOPIC_NODE_ID` 为准：

- `/refresh_grasp_feedback_0_345_{neck,leg_waist,left_arm,right_arm,left_gripper,right_gripper}`：原厂电机进程内只读刷新，协议版本2。
- `/control_independent_sync_hold_session_0_345`：已有保持式SYNC入口。
- `/control_independent_sync_grasp_finish_0_345`：抓取结束时保留夹持目标。
- `/control_independent_sync_loaded_session_0_345`：带物放置/观察的SYNC进入及退出，避免用接触后的反馈开口覆盖夹持目标。

增量文件见 `integration/vendor_20261109/`。其基础版本已经包含既有V5/SYNC适配，不能假定适用于任意原厂SDK。已有345本机已经加载这些扩展，上传源码不修改运行状态。

## 未包含的内容

`models/`中的权重/模型仓库；`data/`中的照片、商品、用户姿态、全身复位目标、录制动作及任务；标定会话、现场地图、缓存、日志；原厂SDK、系统/conda环境和外部ROS包。

需要恢复一套完整环境时，应另外迁移这些资源并核对路径。默认 `geometry_source: urdf` 不依赖手眼标定文件，但不代表不需要本机模型、相机深度内参、夹爪开口映射或可用原厂服务。

## 核对原则

1. 保留既有 `data/`，不要用空目录覆盖现场姿态库与复位目标。
2. 按本机环境调整 `config/grasp.yaml` 的Python、模型、URDF、原厂配置路径；不同机器人不能直接套用345夹爪标定。
3. 分别在系统Python和感知Python验证各自导入，不把所有依赖混装进系统Python。
4. 先核对服务与模型，再做用户现场操作验证。构建或导入成功不自动执行复位/抓取。

本分支保留原始机器人源码和配置，不在发布时额外改动运动逻辑。
