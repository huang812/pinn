import os
import subprocess
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "4 Presentation")
SCRIPT_FOLDER_NAME = "Result_drawing"


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/run_script', methods=['POST'])
def run_script():
    data = request.json
    task = data.get('task')
    data_case = data.get('case')

    if not task or not data_case:
        return jsonify({"status": "error", "message": "缺少任务或数据集参数"}), 400

    script_name = None
    curve_img = None
    error_script_name = None
    error_img = None
    point_error_script_name = None
    point_error_img = None
    img_sub_folder = ""

    # ==========================================
    # 动态分配 3 个脚本及图片名称
    # ==========================================
    if task == "SOH":
        img_sub_folder = "SoH Estimation"
        error_script_name = f"SoH_{data_case}_Error_Bar_Comparison.py"
        error_img = f"SoH_{data_case}_Error_Bar_Comparison.png"
        point_error_img = f"SoH_{data_case}_Pointwise_Absolute_Error_Comparison.png"

        if data_case == "CaseA":
            script_name = "SOH_CaseA.py"
            curve_img = "SoH_CaseA_Prediction.png"
            point_error_script_name = "SOH_A_plot_point_error.py"
        else:
            script_name = f"SOH_{data_case}_Comparison.py"
            curve_img = f"SoH_{data_case}_Prediction_Comparison_Simplified.png"
            point_error_script_name = "SOH_B_plot_point_error.py"

    else:
        img_sub_folder = "RUL Prognostics"
        script_name = f"RUL_{data_case}_Comparison.py"
        error_script_name = f"RUL_{data_case}_Error_Bar_Comparison.py"
        curve_img = f"RUL_{data_case}_Prediction_Comparison.png"
        error_img = f"RUL_{data_case}_Error_Bar_Comparison.png"
        point_error_img = f"RUL_{data_case}_Pointwise_Absolute_Error_Comparison.png"

        if data_case == "CaseA":
            point_error_script_name = "RUL_A_plot_point_error.py"
        else:
            point_error_script_name = "RUL_B_plot_point_error.py"

    logs = []
    metrics_data = []
    cwd_path = os.path.join(BASE_DIR, SCRIPT_FOLDER_NAME)

    try:
        # 1. 执行预测曲线脚本
        if script_name:
            script_path = os.path.join(cwd_path, script_name)
            if not os.path.exists(script_path):
                logs.append(f"❌ 找不到脚本: {script_name}。")
                return jsonify({"status": "error", "message": f"找不到脚本 {script_name}", "logs": logs}), 404

            logs.append(f"正在执行: {script_name}...")
            subprocess.run(["python", script_name], cwd=cwd_path, check=True, capture_output=True, text=True)
            logs.append(f"✅ {script_name} 执行成功！")

        # 2. 执行误差柱状图脚本并提取数据
        if error_script_name:
            error_script_path = os.path.join(cwd_path, error_script_name)
            if not os.path.exists(error_script_path):
                logs.append(f"❌ 找不到误差分析脚本: {error_script_name}。")
                return jsonify({"status": "error", "message": f"找不到脚本 {error_script_name}", "logs": logs}), 404

            logs.append(f"正在执行误差分析: {error_script_name}...")
            process = subprocess.run(["python", error_script_name], cwd=cwd_path, check=True, capture_output=True,
                                     text=True)
            logs.append(f"✅ {error_script_name} 执行成功！误差指标已生成。")

            # 解析终端数据表格
            stdout_text = process.stdout
            for line in stdout_text.split('\n'):
                clean_line = line.replace('%', '').replace(',', '')
                parts = clean_line.strip().split()
                if len(parts) >= 4:
                    try:
                        val1 = float(parts[-3])
                        val2 = float(parts[-2])
                        val3 = float(parts[-1])
                        model_name = " ".join(parts[:-3])
                        if "Model" not in model_name:
                            metrics_data.append({
                                "model": model_name,
                                "rmse": f"{val1:.4f}",
                                "mae": f"{val2:.4f}",
                                "rmspe": f"{val3:.4f}"
                            })
                    except ValueError:
                        pass

        # 3. 🎯 执行逐点误差分析脚本 (核心修复：进入 Point_Error 子文件夹)
        if point_error_script_name:
            point_error_cwd = os.path.join(cwd_path, "Point_Error")
            point_error_script_path = os.path.join(point_error_cwd, point_error_script_name)

            if not os.path.exists(point_error_script_path):
                logs.append(f"❌ 找不到逐点误差脚本: {point_error_script_path}。")
                return jsonify(
                    {"status": "error", "message": f"找不到脚本 {point_error_script_name}，请检查 Point_Error 文件夹",
                     "logs": logs}), 404

            logs.append(f"正在执行逐点误差绘制: {point_error_script_name}...")
            subprocess.run(["python", point_error_script_name], cwd=point_error_cwd, check=True, capture_output=True,
                           text=True)
            logs.append(f"✅ {point_error_script_name} 执行成功！")

    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "message": f"脚本运行出错。详情: {e.stderr}",
            "logs": logs
        }), 500

    return jsonify({
        "status": "success",
        "logs": logs,
        "metrics": metrics_data,
        "images": {
            "curve": f"/images/{img_sub_folder}/{curve_img}" if curve_img else None,
            "error": f"/images/{img_sub_folder}/{error_img}" if error_img else None,
            "point_error": f"/images/{img_sub_folder}/{point_error_img}" if point_error_img else None
        }
    })


# ==========================================
# 🤖 大模型 Q&A 问答接口 (Ollama 兼容版)
# ==========================================
@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"status": "error", "message": "问题不能为空"}), 400

    try:
        ollama_url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "deepseek-r1:1.5b",
            "prompt": user_message,
            "stream": False
        }

        response = requests.post(ollama_url, json=payload)

        if response.status_code != 200:
            error_msg = response.json().get("error", f"HTTP 状态码: {response.status_code}")
            return jsonify({"status": "error", "message": f"后台提示: {error_msg}"}), 500

        result_json = response.json()
        bot_reply = result_json.get("response", "模型无响应内容")

        return jsonify({"status": "success", "reply": bot_reply})

    except requests.exceptions.ConnectionError:
        return jsonify({
            "status": "error",
            "message": "无法连接到大模型后台，请确保你的那个 AI 软件正在运行中！"
        }), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"大模型接口调用出错: {str(e)}"}), 500


@app.route('/images/<path:subpath>')
def serve_image(subpath):
    return send_from_directory(RESULTS_DIR, subpath)


if __name__ == '__main__':
    print("🚀 PINN 分析后端已启动，正在监听 5000 端口...")
    app.run(port=5000, debug=True)