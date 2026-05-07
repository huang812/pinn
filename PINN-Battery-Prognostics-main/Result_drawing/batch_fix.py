import os
import re


def fix_all_files():
    # 获取当前文件夹路径 (Result_drawing)
    current_folder = os.path.dirname(os.path.abspath(__file__))

    # 正则：匹配各种带盘符的写死路径，提取 'Results' 及之后的部分
    pattern_r_path = re.compile(r"r['\"][a-zA-Z]:\\[^\n]*?(Results[\\/][^\n]*?)['\"]", re.IGNORECASE)
    pattern_str_path = re.compile(r"['\"][a-zA-Z]:\\\\[^\n]*?(Results\\\\[^\n]*?)['\"]", re.IGNORECASE)

    count = 0
    print("=" * 50)
    print("🚀 开始批量扫描并修复项目脚本...")
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
            # 1. 自动注入 weights_only=False 参数
            # ----------------------------------------
            def load_repl(match):
                args = match.group(1)
                # 仅在参数里没有 weights_only 时才添加，防止重复添加
                if "weights_only" not in args:
                    return f"torch.load({args}, weights_only=False)"
                return match.group(0)

            # 匹配 torch.load(里面不能有右括号，防止破坏代码结构)
            content = re.sub(r"torch\.load\(([^)]*)\)", load_repl, content)

            # ----------------------------------------
            # 2. 将硬编码绝对路径替换为自动寻址的相对路径
            # ----------------------------------------
            def path_repl(match):
                sub_path = match.group(1).replace('\\\\', '\\').replace('/', '\\')
                parts = sub_path.split('\\')
                parts_str = ", ".join([f"'{p}'" for p in parts if p])
                # 动态退两层到根目录再拼接
                return f"os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), {parts_str})"

            content = pattern_r_path.sub(path_repl, content)
            content = pattern_str_path.sub(path_repl, content)

            # 确保代码顶部有导入 os 模块
            if "os.path.join" in content and "import os" not in content:
                content = "import os\n" + content

            # ----------------------------------------
            # 3. 覆盖写入并反馈
            # ----------------------------------------
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ 已成功修复并保存: {filename}")
                count += 1
            else:
                print(f"➖ 无需修改或此前已修复: {filename}")

    print("=" * 50)
    print(f"🎉 批量修复圆满完成！共帮您修复了 {count} 个文件。")
    print("现在您可以随意运行画图文件夹里的任何脚本了。")


if __name__ == '__main__':
    fix_all_files()