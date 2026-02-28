
PD分离式部署，请参考[使用文档](../../docs/zh/features/disaggregated.md)。

PD分离式部署，推荐使用Router来做请求调度（即是V1模式）。

启动脚本：

* `start_v1_tp1.sh`：使用Router调度，P和D实例是TP1。
* `start_v1_tp2.sh`：使用Router调度，P和D实例是TP2。
* `start_v1_dp2.sh`：使用Router调度，P和D实例是DP2 TP1。
* `start_1e1pd.sh`：非PD分离（单实例 mixed），启动前会自动kill占用端口的进程。
* `start_epd_qwen25vl.sh`：EPD模式（E节点+PD节点+Router），默认模型路径是 `/root/paddlejob/workspace/env_run/output/wangyafeng/models/Qwen2.5-VL-7B-Instruct`，启动前会自动kill占用端口的进程。
* `start_2e1pd_qwen25vl.sh`：EPD模式（2个E节点+1个PD节点+Router），用于2E对比测试，默认模型路径是 `/root/paddlejob/workspace/env_run/output/wangyafeng/models/Qwen2.5-VL-7B-Instruct`。
* `start_pd_only_qwen25vl.sh`：无E节点，对比用单节点 mixed（Router+PD单实例），默认模型路径是 `/root/paddlejob/workspace/env_run/output/wangyafeng/models/Qwen2.5-VL-7B-Instruct`，启动前会自动kill占用端口的进程。

停止脚本：

* `stop_1e1pd.sh`：停止 `start_1e1pd.sh` 启动的服务，并清理相关端口占用。
* `stop_epd_qwen25vl.sh`：停止 `start_epd_qwen25vl.sh` 启动的服务，并清理相关端口占用。
* `stop.sh`：统一停止 splitwise 示例相关服务（包含 `stop_1e1pd.sh` 和 `stop_epd_qwen25vl.sh`）。
