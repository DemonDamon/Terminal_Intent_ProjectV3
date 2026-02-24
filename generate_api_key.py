#!/usr/bin/env python3
"""
企业级 API Key 生成工具
=======================
用于生产环境安全密钥的生成和管理。

功能:
- 生成符合安全规范的强随机密钥
- 支持多种输出格式
- 自动更新 .env 文件

使用方式:
    python generate_api_key.py              # 生成并显示密钥
    python generate_api_key.py --update     # 生成并更新 .env 文件
    python generate_api_key.py --length 64  # 指定密钥长度

安全规范:
- 密钥长度不少于32字符
- 使用密码学安全的随机数生成器
- 包含大小写字母和数字
"""

import secrets
import string
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# 配置
# ============================================================

DEFAULT_KEY_LENGTH = 48  # 默认密钥长度
MIN_KEY_LENGTH = 32      # 最小密钥长度
MAX_KEY_LENGTH = 128     # 最大密钥长度

ENV_FILE = ".env"
ENV_EXAMPLE_FILE = ".env.example"


# ============================================================
# 密钥生成函数
# ============================================================

def generate_secure_api_key(length: int = DEFAULT_KEY_LENGTH) -> str:
    """
    生成密码学安全的API密钥

    Args:
        length: 密钥长度 (默认48字符)

    Returns:
        str: 生成的API密钥
    """
    if length < MIN_KEY_LENGTH:
        raise ValueError(f"密钥长度不能小于 {MIN_KEY_LENGTH} 字符")
    if length > MAX_KEY_LENGTH:
        raise ValueError(f"密钥长度不能大于 {MAX_KEY_LENGTH} 字符")

    # 使用密码学安全的随机数生成器
    # 字符集: 大写字母 + 小写字母 + 数字
    alphabet = string.ascii_letters + string.digits

    # 确保密钥包含至少一个大写、小写和数字
    while True:
        key = ''.join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.isupper() for c in key) and
            any(c.islower() for c in key) and
            any(c.isdigit() for c in key)):
            return key


def generate_hex_api_key(length: int = 32) -> str:
    """
    生成十六进制格式的API密钥

    Args:
        length: 十六进制字符数 (默认32，即16字节)

    Returns:
        str: 十六进制格式的API密钥
    """
    byte_length = length // 2
    return secrets.token_hex(byte_length)


def generate_urlsafe_api_key(length: int = 32) -> str:
    """
    生成URL安全的API密钥

    Args:
        length: 字符数 (默认32)

    Returns:
        str: URL安全的API密钥
    """
    return secrets.token_urlsafe(length)[:length]


# ============================================================
# 文件操作函数
# ============================================================

def update_env_file(api_key: str, env_path: str = ENV_FILE) -> bool:
    """
    更新 .env 文件中的 API_KEY

    Args:
        api_key: 新的API密钥
        env_path: .env 文件路径

    Returns:
        bool: 是否更新成功
    """
    env_file = Path(env_path)

    # 如果 .env 不存在，从 .env.example 复制
    if not env_file.exists():
        example_file = Path(ENV_EXAMPLE_FILE)
        if example_file.exists():
            content = example_file.read_text(encoding='utf-8')
            env_file.write_text(content, encoding='utf-8')
            print(f"[INFO] 从 {ENV_EXAMPLE_FILE} 创建 {env_path}")
        else:
            # 创建最小化的 .env 文件
            env_file.write_text(f"API_KEY={api_key}\n", encoding='utf-8')
            print(f"[INFO] 创建新的 {env_path}")
            return True

    # 读取现有内容并清理Windows换行符
    content = env_file.read_text(encoding='utf-8')
    # 统一换行符为Unix格式
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    lines = content.splitlines()

    # 查找并替换 API_KEY
    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith('API_KEY='):
            new_lines.append(f"API_KEY={api_key}")
            updated = True
        else:
            new_lines.append(line)

    # 如果没有找到 API_KEY，添加到文件开头
    if not updated:
        new_lines.insert(0, f"API_KEY={api_key}")

    # 写回文件
    env_file.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
    return True


def backup_env_file(env_path: str = ENV_FILE) -> str:
    """
    备份 .env 文件

    Args:
        env_path: .env 文件路径

    Returns:
        str: 备份文件路径
    """
    env_file = Path(env_path)
    if env_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{env_path}.backup_{timestamp}"
        env_file.rename(backup_path)
        # 复制回原位置
        Path(backup_path).read_text(encoding='utf-8')
        env_file.write_text(Path(backup_path).read_text(encoding='utf-8'), encoding='utf-8')
        return backup_path
    return ""


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="企业级 API Key 生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_api_key.py                 # 生成并显示密钥
  python generate_api_key.py --update        # 生成并更新 .env 文件
  python generate_api_key.py --length 64     # 指定密钥长度
  python generate_api_key.py --format hex    # 十六进制格式
  python generate_api_key.py --show-only     # 仅显示，不提示更新
        """
    )

    parser.add_argument(
        '--length', '-l',
        type=int,
        default=DEFAULT_KEY_LENGTH,
        help=f"密钥长度 (默认: {DEFAULT_KEY_LENGTH}, 范围: {MIN_KEY_LENGTH}-{MAX_KEY_LENGTH})"
    )

    parser.add_argument(
        '--format', '-f',
        choices=['default', 'hex', 'urlsafe'],
        default='default',
        help="密钥格式 (default: 字母数字混合, hex: 十六进制, urlsafe: URL安全)"
    )

    parser.add_argument(
        '--update', '-u',
        action='store_true',
        help="自动更新 .env 文件"
    )

    parser.add_argument(
        '--backup', '-b',
        action='store_true',
        help="更新前备份 .env 文件"
    )

    parser.add_argument(
        '--show-only', '-s',
        action='store_true',
        help="仅显示密钥，不提示任何操作"
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="静默模式，仅输出密钥"
    )

    args = parser.parse_args()

    # 生成密钥
    try:
        if args.format == 'hex':
            api_key = generate_hex_api_key(args.length)
        elif args.format == 'urlsafe':
            api_key = generate_urlsafe_api_key(args.length)
        else:
            api_key = generate_secure_api_key(args.length)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # 静默模式
    if args.quiet:
        print(api_key)
        sys.exit(0)

    # 显示密钥信息
    print("\n" + "=" * 60)
    print("  企业级 API Key 生成工具")
    print("=" * 60)
    print(f"\n[生成时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[密钥长度] {len(api_key)} 字符")
    print(f"[密钥格式] {args.format}")
    print(f"\n[API_KEY]")
    print("-" * 60)
    print(api_key)
    print("-" * 60)

    # 仅显示模式
    if args.show_only:
        print("\n[提示] 请手动将此密钥配置到生产环境")
        sys.exit(0)

    # 自动更新模式
    if args.update:
        if args.backup:
            backup_path = backup_env_file()
            if backup_path:
                print(f"\n[INFO] 已备份 .env 到 {backup_path}")

        if update_env_file(api_key):
            print(f"\n[SUCCESS] 已更新 .env 文件中的 API_KEY")
        else:
            print(f"\n[ERROR] 更新 .env 文件失败", file=sys.stderr)
            sys.exit(1)
    else:
        # 交互式提示
        print("\n[操作选项]")
        print("  1. 手动复制上述密钥到 .env 文件")
        print("  2. 运行 'python generate_api_key.py --update' 自动更新")
        print("  3. 设置环境变量: export API_KEY=<密钥>")

    # 安全提示
    print("\n[安全提示]")
    print("  - 请勿在代码中硬编码此密钥")
    print("  - 请勿将密钥提交到版本控制系统")
    print("  - 建议定期轮换生产环境密钥")
    print("  - 请妥善保管此密钥，丢失后需重新生成")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
