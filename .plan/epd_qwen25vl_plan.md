# FastDeploy Qwen2.5VL EPD 分离（E+PD）实施计划

## 1. 目标摘要
- 在 FastDeploy 中实现面向 `Qwen2.5VL` 的 EPD 分离首版。
- `E 节点`负责 `processor + ViT`，通过本机 `IPC SHM` 将视觉特征传递给 `PD 节点`。
- `PD 节点`负责 prefill/decode，并复用现有 splitwise/请求调度主干。
- 同时交付 `1e1pd（非PD分离）` 启动脚本与 stop 脚本。

## 2. 范围与非目标
- 首版范围：
  - 单机同宿主（`/dev/shm`）进程间传输。
  - 模型仅支持 `Qwen2.5VL`。
  - 优先支持 GPU 路径。
- 非目标：
  - 跨机 E->PD 特征传输。
  - 一次性覆盖所有后端（metax/xpu/hpu/gcu）。
  - 非 Qwen2.5VL 的多模态模型适配。

## 3. 外部接口与配置变更
1. 扩展 `splitwise_role`：
   - 现有：`mixed | prefill | decode`
   - 新增：`encoder`
2. 新增 EPD 开关与参数：
   - `--epd-enable`（bool，默认 false）
   - `--epd-shm-dir`（默认 `/dev/shm`）
   - `--epd-shm-ttl-sec`（默认 120）
   - `--epd-shm-max-bytes`（默认 4GB）
3. 请求/协议字段（新增可选）：
   - `vision_shm_refs`
   - `vision_meta`（shape/dtype/grid_thw/request_id/chunk_id 等）
4. 兼容策略：
   - `epd-enable=false` 时，行为与当前实现保持一致。

## 4. 架构与数据流
1. `encoder` 进程收到请求，执行 processor 与 ViT，得到 `image_features_list`。
2. `encoder` 将特征写入 SHM，产出元数据 `VisionShmRef`：
   - `shm_name, offset, nbytes, shape, dtype, request_id`
3. `encoder` 通过现有控制面（splitwise connector）仅发送 metadata，不发送大 tensor。
4. `pd` 进程收到请求后，按 `VisionShmRef` attach SHM 并恢复 tensor，注入 prefill 输入构造流程。
5. decode 继续走现有路径（含 `position_ids` 处理），不重复提取 ViT 特征。
6. 生命周期控制：
   - E 创建 + 登记 shm；
   - PD attach 成功后 ACK；
   - E 在 ACK 或超时后释放；
   - 后台 TTL 清理器回收孤儿段，防止 `/dev/shm` 泄漏。

## 5. 代码改造清单

### 5.1 参数与配置层
- 修改：
  - `fastdeploy/config.py`
  - `fastdeploy/engine/arg_utils.py`（或对应参数定义文件）
- 内容：
  - 支持 `splitwise_role=encoder`
  - 注入 EPD 新参数与校验

### 5.2 协议与连接层
- 修改：
  - `fastdeploy/splitwise/splitwise_connector.py`
  - 请求消息结构定义相关文件
- 内容：
  - 新增 `encoder -> pd` 消息类型（如 `encoder2prefill`）
  - 在请求载荷中支持 `vision_shm_refs/vision_meta`

### 5.3 IPC SHM 管理层
- 新增/修改：
  - `fastdeploy/inter_communicator/` 下新增 `epd_shm_manager.py`（或等价模块）
- 能力：
  - `create/attach/release/refcount/ttl_gc`
  - 段命名规则、大小限制、过期回收与异常清理

### 5.4 Worker 路径（先 GPU）
- 修改：
  - `fastdeploy/worker/gpu_model_runner.py`
- 内容：
  - `encoder` 角色：执行 `_process_mm_features`，将结果写 SHM 并回传 ref
  - `prefill/decode`：若带 `vision_shm_refs`，走 SHM 反序列化分支
  - 保留无 EPD 时原逻辑

### 5.5 Qwen2.5VL 适配
- 重点：
  - 尽量不改 `qwen2_5_vl.py` 主干行为
  - 在进入模型前构造好 `image_features`，沿现有 embedding 注入路径工作

### 5.6 路由与调度
- 修改相关 router/scheduler 文件：
  - 增加 `encoder -> pd` 转发链路与状态机钩子
  - 增加超时、attach 失败、metadata 不一致等错误处理

## 6. 日志设计（必须可观测）
- 统一前缀：
  - `[EPD][E]`
  - `[EPD][PD]`
  - `[EPD][SHM]`
  - `[EPD][ROUTER]`
- 关键日志字段：
  - `request_id, batch_size, feature_bytes, shm_name`
  - `create/attach/ack/release` 时延
  - 异常原因与错误码
- 目标：
  - 能通过日志快速判断请求卡在 E、IPC、PD 哪一段。

## 7. 脚本交付
1. 新增：`scripts/epd/start_1e1pd.sh`
   - 启动单进程（`splitwise_role=mixed`）
   - 显式 `epd-enable=false`
   - 输出 PID 与日志路径
2. 新增：`scripts/epd/stop_1e1pd.sh`
   - 读取 PID 文件优雅停止，超时后强制 kill
   - 清理 pid/lock 运行时文件
   - 幂等处理（重复执行不报错）

## 8. 测试与验收
1. 功能正确性：
   - Qwen2.5VL 单图/多图请求在 EPD 与非 EPD 下结果一致（允许浮点微差）。
2. 稳定性：
   - 长压测下 SHM 无持续泄漏。
3. 故障注入：
   - PD attach 超时、E 异常退出、非法 shm_name、shape 不匹配均可被识别并安全失败。
4. 回归：
   - `epd-enable=false` 下文本与现有多模态主路径不受影响。
5. 脚本：
   - `start_1e1pd.sh` 可重复执行并检测重复启动。
   - `stop_1e1pd.sh` 幂等。

## 9. 实施顺序
1. 参数/配置/角色扩展。
2. SHM 管理器与协议字段打通。
3. GPU worker 的 encoder 与 pd 接入。
4. router/scheduler 链路联调。
5. 日志补齐与故障处理。
6. 脚本交付与测试收敛。

## 10. 默认假设
- 只支持单机同宿主 EPD。
- 首版仅保证 GPU + Qwen2.5VL。
- 其他后端首版可显式 `NotImplemented` 并打印清晰日志。
