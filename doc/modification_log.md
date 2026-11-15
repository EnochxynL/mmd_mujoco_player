# 收到，就停在这里，代码保持现在的状态（已改完并验证过，未再改动）。

## 结论：不是"左右臂参数写反"，而是四个叠在一起的坐标/绑骨缺陷

用你截图里的那帧做定量复现后，`before` 左边那张确实逐像素重现了你发的图（两手在裆前交叉成 X）。原因：

1. **手性没转换（真正的"左右反了"）**
   MMD/PMX 是左手系，MuJoCo 是右手系，二者之间必须含**一次反射**（det = −1）。原代码 `_MMD_TO_MUJOCO_Q = [0.5,0.5,0.5,0.5]` 只是绕 (1,1,1) 转 120°，det = +1——四元数本身只能编码纯旋转，所以**每一个旋转都被镜像了**，外展变成内收。
   证据：原生成器 `(x,y,z)→(z,x,y)` 造出的隐性骨架**脚尖朝 −X**（实测左 `toe−ankle = [-0.126, 0, -0.013]`），而 G1 朝 +X——整套骨架是 G1 的镜像。

2. **rest pose 不一致（交叉的直接主因）**
   PMX 的 rest 是 A-pose（大臂约 40° 外展），G1 的零位是**双臂自然下垂**。VMD 的旋转是相对 PMX rest 的增量，直接扣到 G1 上就等于多转了 ~50°，手被甩过垂直线后折向体前 → 裆前交叉。实测大臂方向误差 **52°/58°**。

3. **`straight_q` 是猜的**
   实测"肘伸直"在 **q = 73°**（不是 90°），"膝伸直"在 **q = 14°**（不是 0°），所以前臂/小腿永远带 17°/14° 系统误差。

4. **`センター` 的旋转被整个丢掉**（只取了它的位移）
   第 240 帧 `センター` 转了 72°，机器人却原地不动。`センター` 是 PMX 里骨盆的父节点，必须并进 floating base。

## 改了什么

- `src/mmd_robot_player/player.py`
  ```python
  q = np.array([w, z, -x, -y])            # 轴置换 + 手性翻转
  pos = scale * np.array([-z, x, y])      # 同一基变换
  ```
  新增 `bind_targets / bind_all / update_pose`：每帧把 ball joint 的 rest 旋转除掉（MuJoCo 是 `parent·rest·q`，PMX 是相对无旋转父骨），再乘一个**rest 对齐常量**（把 G1 零位肢体方向旋到 PMX rest 方向，实测大臂 49.7°/50.4°、大腿 9.9°/10.5°），并把 `全ての親·センター` 叠进 free joint。
- `src/mmd_robot_player/robots.py`：新增 `segment_align`，`straight_q` 改成实测值 73°/14°。
- `tools/build_g1_pmx_mjcf.py`：`pmx_to_mjcf` 改用 `(-z, x, y)`；已按同一等价变换（镜像 YZ 平面，`pmx_body_*` 的局部 x 取负）修好随仓库分发的 `g1_pmx_fk.xml` + `pmx_skeleton.json`，**不需要 PMX 原件重新生成**。
- `tests/test_retarget.py`：新增 6 条回归（手性语义、位移映射、骨架朝向、0 帧双手不交叉、大臂跟随源骨架、根节点跟随 `センター`）。

## 验证（同一相机、逐帧对照源 PMX 骨架 FK）

| | 修复前 | 修复后 |
|---|---|---|
| 0 帧左手 y / 右手 y | −0.039 / +0.027（交叉） | **+0.209 / −0.213**（各自在正确一侧） |
| 大臂方向误差 左/右 | 52° / 58° | **0.0° / 0.0°** |
| 前臂方向误差 左/右 | 57° / 70° | **4.4° / 8.8°**（全帧 1.5°–17°） |
| 双手间距 | 0.06 m | 0.42 m |

`uv run python -m unittest discover -s tests` → **7 tests OK**。上面那张对比图：上排第 0 帧、下排第 120 帧，**左=修复前、右=修复后**（图在 `C:\Users\enoch\AppData\Local\Temp\mmd_diag\compare_before_after.png`）。

## 仍存的限制（不是这次能靠调号解决的）

- 肘/腕是 1 自由度近似：`apply_flexion` 只用源骨架的弯曲**幅值**，弯曲平面由 `axis` 固定，所以前臂仍有 ≤17° 残差。
- PMX 的 IK、付与親（肩C/腕捩在这一版里旋转恰好为 0，所以大臂能对到 0°）、非根位移仍未实现；`左腕捩→wrist_roll` 的语义映射仍是权宜。
- README 的「当前边界 / 坐标映射」段还写着旧映射（`MMD [x,y,z] -> MuJoCo [z,x,y]`），要改的话你说一声，我不动它。

# 已按右边（融合后）为准提交。

```
0b8f4fd  Keep the G1 humerus rigid instead of tearing it across three links
a539a5b  doc
e0985c8  Fix MMD->MuJoCo handedness, rest-pose offset and missing root rotation
```

工作区干净，8 tests OK。

## 问题定位

G1 的肱骨分在 `shoulder_pitch → shoulder_roll → shoulder_yaw` 三个 link 上，MMD 只有一根 `腕`。旧映射把 `腕` 放在 `shoulder_yaw`（下段），再叠加 ~50° 的 rest 对齐（也只落在这一个 ball joint 上），于是**上段两截不动、下段整根转**——就是你看到的断裂。

## 改法

- `robots.py`：`肩P / 肩 / 腕` 三根 PMX 骨全部 compose 到 **`shoulder_pitch_joint`**（链的最上端），`shoulder_roll/yaw` 保持零位 → 整根肱骨刚性旋转；`segment_align` 移到 `shoulder_pitch`，并用新增的 `g1_start` 指定"量哪一段"（量 `shoulder_yaw → elbow` 这根真正的肱骨，而不是从 pitch 量）。
- `player.py`：`bind_targets` 改成两遍——先按 rest 反解算出"所有 delta = identity"时的实际姿态（probe），再在这个姿态上量段方向求对齐角；否则 pitch 关节自身的 rest 旋转被除掉后，量出来的段并不是模型原始朝向，角度会偏。
- `tests/test_retarget.py`：新增 `test_humerus_stays_rigid`，三个 link 彼此的相对朝向偏离 rest > 0.5° 就失败。

## 数值

| | 改前（distributed） | 改后（fused） |
|---|---|---|
| 肱骨方向 vs PMX | 0.0° | **0.0°**（不变，合成旋转相同） |
| 三段相对朝向 | pitch→roll 差 ~14°、roll→yaw 差 ~50°+36° | **全 0°（刚性）** |
| 对齐角 | 49.7°（挂在 yaw 上） | 65.4°（挂在 pitch 上，按其自身 rest 反解后折算） |
| 0 帧双手 y | +0.209 / −0.213 | +0.24 / −0.24 |
| 前臂误差 | 3.8°/8.1° | 3.8°/8.1°（不变） |

## 两个要留意的副作用

1. **手的位置整体外移约 3 cm**（y 0.21 → 0.24）。刚性化后肱骨以肩关节为支点整体旋转，三段不再各自微转，肘/手的绝对落点会略有变化——方向仍是 PMX 的精确值。
2. **`shoulder_roll` / `shoulder_yaw` 现在不参与驱动**（保持零位）。这是视觉播放的取舍：真机那三根 DOF 的分工要靠 mesh 的滑移壳来容纳，而 MMD 的 50° A-pose 补偿压在任一子关节上都会撕网。以后若要做动力学轨迹（需要真实三自由度分配），得换成型面/世界矩阵 retarget，而不是直接复用这份 qpos。