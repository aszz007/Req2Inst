# scripts/run_with_env.py
"""
环境管理脚本：根据任务类型自动切换环境
解决Windows中文乱码问题
"""
import subprocess
import sys
import argparse
import os

# 设置环境变量以解决Windows中文编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 在Windows上设置控制台为UTF-8模式
if sys.platform == 'win32':
    try:
        # 设置控制台代码页为UTF-8
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
    except:
        pass

# 环境映射
ENV_MAP = {
    'text': 'qwen_text',
    'image_qwen2.5': 'qwen_vision25',
    'image_qwen3': 'qwen_vision3',
    'uml_qwen2.5': 'qwen_vision25',
    'uml_qwen3': 'qwen_vision3',
}


def run_in_env(env_name: str, script_path: str, args: list = None):
    """在指定环境中运行脚本"""

    # 从环境名推断Qwen版本
    qwen_version = None
    if 'qwen3' in env_name or env_name == 'qwen_vision3':
        qwen_version = 'qwen3'
    elif 'qwen2.5' in env_name or 'qwen25' in env_name or env_name == 'qwen_vision25':
        qwen_version = 'qwen2.5'

    # 初始化参数列表
    if args is None:
        args = []

    # 自动添加 --version 参数（如果推断出了版本且参数中没有 --version）
    if qwen_version and '--version' not in args:
        args = ['--version', qwen_version] + args

    # 构建conda命令
    cmd = [
        'conda', 'run',
        '-n', env_name,
        '--no-capture-output',  # 关键：避免conda捕获输出导致编码问题
        'python', script_path
    ]

    if args:
        cmd.extend(args)

    print(f"运行环境: {env_name}")
    if qwen_version:
        print(f"Qwen版本: {qwen_version}")
    print(f"脚本: {script_path}")
    print(f"参数: {' '.join(args) if args else '无'}")
    print("-" * 60)

    # 准备环境变量（复制当前环境变量并添加QWEN_VISION_VERSION）
    env = os.environ.copy()
    if qwen_version:
        env['QWEN_VISION_VERSION'] = qwen_version

    # 执行命令，传递环境变量
    result = subprocess.run(cmd, env=env)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='环境管理脚本：自动切换Conda环境执行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 运行文本任务
  python scripts/run_with_env.py --env text --script scripts/training/train_text_expert.py

  # 运行图像识别（Qwen2.5）
  python scripts/run_with_env.py --env image_qwen2.5 --script models/vision_model.py inputs/image/test.jpg

  # 运行UML识别（Qwen3）
  python scripts/run_with_env.py --env uml_qwen3 --script scripts/training/train_uml_expert.py
        """
    )

    parser.add_argument(
        '--env',
        type=str,
        required=True,
        choices=list(ENV_MAP.keys()),
        help='环境类型'
    )

    parser.add_argument(
        '--script',
        type=str,
        required=True,
        help='要执行的脚本路径'
    )

    parser.add_argument(
        'script_args',
        nargs='*',
        help='传递给脚本的参数（可选）'
    )

    args = parser.parse_args()

    env_name = ENV_MAP[args.env]
    exit_code = run_in_env(env_name, args.script, args.script_args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()