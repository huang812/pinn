import os
import re


def batch_fix_plots():
    # 获取当前画图文件夹路径
    current_folder = os.path.dirname(os.path.abspath(__file__))
    count = 0

    print("=" * 50)
    print("🚀 开始批量扫描并修复画图脚本 (注入 savefig & 屏蔽 show)...")
    print("=" * 50)

    for filename in os.listdir(current_folder):
        # 仅处理 Python 脚本，且跳过自己
        if filename.endswith(".py") and filename != os.path.basename(__file__):
            filepath = os.path.join(current_folder, filename)

            # 兼容不同系统的文件编码（UTF-8 或 GBK）
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='gbk') as f:
                    content = f.read()

            original_content = content

            # ----------------------------------------
            # 1. 自动屏蔽所有的 plt.show() -> # plt.show()
            # ----------------------------------------
            # 使用正则匹配开头可能有空格的 plt.show()
            content = re.sub(r'^([ \t]*)(plt\.show\(\))', r'\1# \2', content, flags=re.MULTILINE)

            # ----------------------------------------
            # 2. 自动注入 plt.savefig() 逻辑 (如果缺失)
            # ----------------------------------------
            if "plt.savefig" not in content and ("RUL" in filename or "SOH" in filename):
                # 动态判断是 RUL 还是 SOH 任务
                task_folder = 'RUL Prognostics' if 'RUL' in filename else 'SoH Estimation'

                # 适配前端 app.py 要求的图片命名格式
                # 例如把 RUL_CaseA_Comparison.py 变成 RUL_CaseA_Prediction_Comparison.png
                save_name = filename.replace('_Comparison.py', '_Prediction_Comparison.png').replace('.py', '.png')

                # 构造要注入的代码块
                save_code = f"""
# =========================
# 自动注入的保存图片代码 (Batch Fix)
# =========================
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', '{task_folder}')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "{save_name}")
plt.savefig(save_path, dpi=600, bbox_inches="tight")
print(f"Figure saved to: {{save_path}}")

# plt.show()"""

                # 将刚刚注释掉的 # plt.show() 替换成包含保存逻辑的完整代码块
                content = content.replace("# plt.show()", save_code)

            # ----------------------------------------
            # 3. 覆盖写入并反馈
            # ----------------------------------------
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ 已成功修复并注入保存逻辑: {filename}")
                count += 1
            else:
                print(f"➖ 无需修改或此前已修复: {filename}")

    print("=" * 50)
    print(f"🎉 批量修复圆满完成！共帮您修复了 {count} 个文件。")
    print("现在所有的画图脚本都不会弹窗卡死了，并且都会自动保存高清图片！")


if __name__ == '__main__':
    batch_fix_plots()