# PMX → MuJoCo FK → VMD 播放

这个项目只做两件事：

1. 从 PMX 提取骨骼名称、父子层级和 rest pose 位置，把它们作为无几何的球关节骨架加入 G1 MJCF。
2. 解析 VMD 骨骼轨道，把局部旋转写入这些 MuJoCo 球关节，用 `mj_forward` 计算 FK 并播放。

可见 G1 的 STL、刚体层级、body 位姿与 geom 定义保留原样。新增 PMX 骨架没有 geom，不会改变 G1 建模；目标 G1 关节改为 ball joint，以便接收三维旋转。当前模型有 122 个 PMX 隐性关节和 29 个 G1 关节。

## 1. PMX 骨架转成 MJCF

先准备 Unitree G1 STL 资源包，并将其 XML 基础模型与 `assets/` 放在同一目录。本项目中基础文件为 `models/unitree_g1/xmls/g1_visual_base.xml`。然后运行：

```bash
python tools/build_g1_pmx_mjcf.py \
  --g1-xml models/unitree_g1/xmls/g1_visual_base.xml \
  --pmx /path/to/source_model.pmx \
  --output models/unitree_g1/xmls/g1_pmx_fk.xml \
  --map-output models/unitree_g1/pmx_skeleton.json
```

`--scale` 控制 PMX 长度单位到米的换算，默认为 `0.07`。生成器会把 PMX 骨骼表写成 MJCF 的无几何球关节层级，并保存骨名到 joint 名的 JSON 映射。它只读取 PMX 的骨骼元数据，不依赖 `pypmxvmd`。

## 2. 在 MJCF 上播放 VMD

```bash
uv sync
uv run mmd-player motions/胡桃待機モーション.vmd --loop
```

默认使用上一步生成的 G1-PMX MJCF 和映射文件。也可以指定其他生成好的模型与映射：

```bash
uv run mmd-player /path/to/motion.vmd \
  --xml /path/to/model_with_pmx_fk.xml \
  --skeleton /path/to/pmx_skeleton.json \
  --speed 0.8 --loop
```

用 `--list` 列出 VMD 中的骨骼轨道；不启动 viewer：

```bash
uv run mmd-player motions/胡桃待機モーション.vmd --list
```

## 结构

```text
models/unitree_g1/xmls/g1_visual_base.xml   G1 原始视觉/刚体模型
models/unitree_g1/xmls/g1_pmx_fk.xml        加入 PMX FK 骨架后的模型
models/unitree_g1/pmx_skeleton.json         PMX 骨名与隐性 joint 映射
src/mmd_robot_player/vmd.py                 VMD 解析与关键帧插值
src/mmd_robot_player/player.py              MuJoCo FK 播放器
src/mmd_robot_player/robots.py              G1 根节点和目标关节映射
tools/build_g1_pmx_mjcf.py                  PMX 骨架 MJCF 生成器
```

新增机器人时，只需准备其视觉/刚体 MJCF，并在 `robots.py` 中维护该机器人的 XML、根节点和目标关节映射；VMD 读取、插值和 FK 播放逻辑保持不变。

## 当前边界

- 肘、膝采用源 PMX 骨架 FK 的三点夹角驱动：按 G1 的局部 Y 轴、伸直零位和屈曲范围生成姿态，避免反屈及侧向折弯。校准参数集中在 `robots.py` 的 `flexion` 字典中。此处保留弯曲幅度，弯曲平面由 G1 配置决定；源隐性骨架四元数不受影响。这仍是几何近似，尚未实现全身 IK。

- 隐性骨架保留 PMX 的骨名、父子关系、位置和每根骨骼的三维旋转自由度；PMX 局部轴、骨骼继承、IK 约束以及非根骨骼平移尚未实现。
- 目前只有躯干、肩臂、肘、腕和腿部的部分标准骨骼会直映到可见 G1；这不是 IK retarget。其他匹配骨骼仍在隐性骨架上参与 FK，但不会驱动 G1 STL。
- 不运行重力、碰撞或动力学仿真。G1 手指、面部和固定头部几何不会因未映射的骨骼轨道变形。
