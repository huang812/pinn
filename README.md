# 🔋 PINN-Battery-Prognostics: 电化学储能电池状态监测与分析系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-green.svg)
![Ollama](https://img.shields.io/badge/AI_Agent-Ollama(DeepSeek)-orange.svg)
![Institution](https://img.shields.io/badge/Institution-NCEPU_%26_ICRC-003b8e.svg)

基于物理信息神经网络（PINN）与 LLM 协同驱动的电化学储能电池智能监测与分析平台。本项目旨在为微电网、共享储能及电动汽车（EV）场景下的高能量密度电池组提供高精度的健康状态（SOH）评估与剩余寿命（RUL）快速推演，并提供学术级的可视化及 AI 专家辅助排错。

## ✨ 核心功能 (Features)

* **⚙️ 核心模型自动化推演**：支持一键下发计算任务，自动化调度后端 Python 推理引擎，执行 **PINN-Verhulst**（SOH 评估）与 **GRU-DeepHPM**（RUL 预测）物理模型计算。
* **📈 学术级动态可视化**：自动捕获并提取终端预测评估指标（RMSE, MAE, RMSPE），并实时渲染输出高精度对比图谱（预测曲线、误差柱状图、逐点绝对误差图）。
* **🤖 本地 AI 专家助手**：深度集成 Ollama 本地推理引擎（默认接入 `deepseek-r1:1.5b`），提供代码级报错排查、电化学机理归纳及实验数据辅助分析，全程保障科研数据隐私。
* **🖥️ 现代化沙盘 UI**：采用无冗余依赖的 HTML/CSS/JS 原生开发，具备“沙盘门户 - 功能内页”双层架构，交互流畅，支持控制台实时日志输出。

## 📂 项目结构 (Project Structure)

```text
📦 PINN-Battery-Prognostics
 ┣ 📂 Results                 # 分析结果与图表存储目录
 ┃ ┗ 📂 4 Presentation
 ┃   ┣ 📂 RUL Prognostics     # RUL 预测生成的曲线与误差图
 ┃   ┗ 📂 SoH Estimation      # SOH 评估生成的曲线与误差图
 ┣ 📂 Result_drawing          # 核心绘图与模型计算脚本目录
 ┃ ┣ 📂 Point_Error           # 逐点绝对误差计算脚本专用目录
 ┃ ┣ 📜 SOH_CaseA.py          # SOH Case A 计算脚本
 ┃ ┣ 📜 RUL_CaseA_Comparison.py # RUL Case A 计算脚本
 ┃ ┗ 📜 ... (其他推演与误差评估脚本)
 ┣ 📂 static                  # 静态资源目录 (Logo 等)
 ┣ 📜 app.py                  # Flask 后端主程序 (提供路由、脚本文档执行与 LLM 接口)
 ┣ 📜 index.html              # 前端交互界面
 ┗ 📜 README.md               # 项目说明文档
🛠️ 环境依赖与安装指引 (Installation)
系统推荐运行于 Windows 环境下，并使用 Conda 进行环境隔离（适配 4G 显存设备的本地轻量化部署）。

1. 基础环境配置
建议创建一个新的 Conda 虚拟环境以避免依赖冲突：

Bash
# 创建并激活 Conda 环境
conda create -n socgpu python=3.9
conda activate socgpu

# 安装后端框架依赖
pip install Flask flask-cors requests
(注意：请确保该环境中已安装你底层脚本 Result_drawing 所需的科学计算库，如 torch, numpy, matplotlib, pandas 等。)

2. 部署本地大语言模型 (Ollama)
本项目 AI 专家助手依赖本地运行的 Ollama 服务：

前往 Ollama 官网 下载并安装 Windows 版本。

打开终端，拉取并运行轻量级问答模型（适配 RTX 3050 等 4G 显存设备）：

Bash
ollama run deepseek-r1:1.5b
确保 Ollama 默认运行在 http://127.0.0.1:11434。

🚀 启动运行 (Usage)
确保本地大模型后台（Ollama）处于运行状态。

在项目根目录下，启动 Flask 后端服务器：

Bash
python app.py
看到 🚀 PINN 分析后端已启动，正在监听 5000 端口... 提示后，打开浏览器访问：
👉 http://127.0.0.1:5000

模块使用指南
系统总览：查看当前架构与资源池。

模型推演：在下拉框选择 SOH 或 RUL 任务，以及对应的基准/动态工况（Case A / Case B），点击启动引擎，系统将在控制台打印实时执行日志。

可视化结果：推演完成后，该页面将自动刷新，展示多模型对比的性能评估报告。

专家助手：在问答框中输入关于代码排错（如 gurobipy 配置问题）或学术概念的提问，与本地 Agent 实时交流。
