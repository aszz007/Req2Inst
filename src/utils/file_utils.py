"""
文件操作工具集
功能：提供统一的文件、路径、JSON、CSV操作接口
特性：
  - 跨平台路径处理
  - 安全的文件读写
  - JSON/CSV批量处理
  - 模型权重管理
  - 错误处理和日志记录
作者：File Utils System
日期：2025-01-23
"""

import json
import csv
import shutil
import os
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Callable, Iterator
import warnings
from datetime import datetime


# ===== 路径操作 =====

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path: 目录的Path对象

    Example:
        >>> ensure_dir('outputs/results')
        PosixPath('outputs/results')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_path_join(*paths: Union[str, Path]) -> Path:
    """
    安全的跨平台路径拼接

    Args:
        *paths: 要拼接的路径片段

    Returns:
        Path: 拼接后的Path对象

    Example:
        >>> safe_path_join('data', 'raw', 'images')
        PosixPath('data/raw/images')
    """
    if not paths:
        return Path('.')

    result = Path(paths[0])
    for p in paths[1:]:
        result = result / p

    return result


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
    """
    获取相对路径

    Args:
        path: 目标路径
        base: 基准路径

    Returns:
        Path: 相对路径

    Example:
        >>> get_relative_path('/home/user/project/data', '/home/user/project')
        PosixPath('data')
    """
    path = Path(path).resolve()
    base = Path(base).resolve()

    try:
        return path.relative_to(base)
    except ValueError:
        # 如果无法计算相对路径，返回绝对路径
        return path


def validate_path_exists(
        path: Union[str, Path],
        path_type: str = 'auto',
        raise_error: bool = True
) -> bool:
    """
    验证路径是否存在

    Args:
        path: 要验证的路径
        path_type: 路径类型 ('file', 'dir', 'auto')
        raise_error: 如果路径不存在是否抛出异常

    Returns:
        bool: 路径是否存在

    Raises:
        FileNotFoundError: 当路径不存在且raise_error=True时
    """
    path = Path(path)

    # 检查路径是否存在
    if not path.exists():
        if raise_error:
            raise FileNotFoundError(f"路径不存在: {path}")
        return False

    # 检查路径类型
    if path_type == 'file' and not path.is_file():
        if raise_error:
            raise ValueError(f"期望文件，但路径是目录: {path}")
        return False

    if path_type == 'dir' and not path.is_dir():
        if raise_error:
            raise ValueError(f"期望目录，但路径是文件: {path}")
        return False

    return True


# ===== JSON操作 =====

def load_json(filepath: Union[str, Path], encoding: str = 'utf-8') -> Dict:
    """
    加载JSON文件

    Args:
        filepath: JSON文件路径
        encoding: 文件编码

    Returns:
        dict: JSON数据

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    filepath = Path(filepath)
    validate_path_exists(filepath, path_type='file')

    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"JSON格式错误 ({filepath}): {str(e)}",
            e.doc, e.pos
        )


def save_json(
        data: Dict,
        filepath: Union[str, Path],
        indent: int = 2,
        encoding: str = 'utf-8',
        ensure_ascii: bool = False
) -> None:
    """
    保存JSON文件

    Args:
        data: 要保存的数据
        filepath: 保存路径
        indent: 缩进空格数
        encoding: 文件编码
        ensure_ascii: 是否转义非ASCII字符
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'w', encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def update_json(
        filepath: Union[str, Path],
        updates: Dict,
        create_if_missing: bool = True
) -> Dict:
    """
    更新JSON文件（合并字典）

    Args:
        filepath: JSON文件路径
        updates: 要更新的数据
        create_if_missing: 如果文件不存在是否创建

    Returns:
        dict: 更新后的完整数据
    """
    filepath = Path(filepath)

    # 加载现有数据
    if filepath.exists():
        data = load_json(filepath)
    elif create_if_missing:
        data = {}
    else:
        raise FileNotFoundError(f"JSON文件不存在: {filepath}")

    # 合并数据
    data.update(updates)

    # 保存
    save_json(data, filepath)

    return data


# ===== CSV操作 =====

def load_csv(
        filepath: Union[str, Path],
        encoding: str = 'utf-8',
        delimiter: str = ',',
        skip_header: bool = False
) -> List[Dict]:
    """
    加载CSV文件为字典列表

    Args:
        filepath: CSV文件路径
        encoding: 文件编码
        delimiter: 分隔符
        skip_header: 是否跳过标题行

    Returns:
        list: 字典列表，每个字典代表一行
    """
    filepath = Path(filepath)
    validate_path_exists(filepath, path_type='file')

    with open(filepath, 'r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        if skip_header:
            next(reader, None)

        return list(reader)


def load_csv_chunks(
        filepath: Union[str, Path],
        chunksize: int = 1000,
        encoding: str = 'utf-8',
        delimiter: str = ','
) -> Iterator[List[Dict]]:
    """
    分块读取大型CSV文件（生成器）

    Args:
        filepath: CSV文件路径
        chunksize: 每块行数
        encoding: 文件编码
        delimiter: 分隔符

    Yields:
        list: 每块数据（字典列表）

    Example:
        >>> for chunk in load_csv_chunks('large_file.csv', chunksize=1000):
        ...     process(chunk)
    """
    filepath = Path(filepath)
    validate_path_exists(filepath, path_type='file')

    with open(filepath, 'r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        chunk = []
        for i, row in enumerate(reader):
            chunk.append(row)

            if (i + 1) % chunksize == 0:
                yield chunk
                chunk = []

        # 最后一块（可能不足chunksize）
        if chunk:
            yield chunk


def save_csv(
        data: List[Dict],
        filepath: Union[str, Path],
        fieldnames: Optional[List[str]] = None,
        encoding: str = 'utf-8',
        delimiter: str = ','
) -> None:
    """
    保存数据为CSV文件

    Args:
        data: 要保存的数据（字典列表）
        filepath: 保存路径
        fieldnames: 列名（如果为None则从第一行数据推断）
        encoding: 文件编码
        delimiter: 分隔符
    """
    if not data:
        warnings.warn("保存的数据为空")
        return

    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    # 推断fieldnames
    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(filepath, 'w', encoding=encoding, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)


# ===== 模型权重操作 =====

def load_lora_weights(expert_name: str) -> Optional[Path]:
    """
    加载LoRA权重路径

    Args:
        expert_name: 专家名称 ('text', 'image', 'uml', 'general')

    Returns:
        Path: LoRA权重目录路径，如果不存在返回None
    """
    try:
        from config import get_path_config
        path_cfg = get_path_config()
        weight_path = path_cfg.get_expert_weight_path(expert_name)

        if weight_path.exists():
            return weight_path
        else:
            warnings.warn(f"{expert_name}专家的LoRA权重未找到: {weight_path}")
            return None
    except ImportError:
        warnings.warn("配置模块未加载，无法获取权重路径")
        return None


def save_lora_weights(
        model: Any,
        expert_name: str,
        checkpoint_name: Optional[str] = None,
        save_method: str = 'peft'
) -> Path:
    """
    保存LoRA权重

    Args:
        model: 模型对象（PEFT模型）
        expert_name: 专家名称
        checkpoint_name: checkpoint名称（如果为None则使用时间戳）
        save_method: 保存方法 ('peft' 或 'custom')

    Returns:
        Path: 保存路径
    """
    try:
        from config import get_path_config
        path_cfg = get_path_config()

        # 确定保存目录
        weight_path = path_cfg.get_expert_weight_path(expert_name)
        ensure_dir(weight_path)

        # 生成checkpoint名称
        if checkpoint_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_name = f"checkpoint_{timestamp}"

        save_path = weight_path / checkpoint_name
        ensure_dir(save_path)

        # 保存权重
        if save_method == 'peft':
            model.save_pretrained(save_path)
        else:
            # 自定义保存逻辑
            import torch
            torch.save(model.state_dict(), save_path / "model.pt")

        return save_path

    except Exception as e:
        raise RuntimeError(f"保存LoRA权重失败: {str(e)}")


def list_checkpoints(expert_name: str) -> List[Path]:
    """
    列出指定专家的所有checkpoint

    Args:
        expert_name: 专家名称

    Returns:
        list: checkpoint路径列表（按时间排序）
    """
    try:
        from config import get_path_config
        path_cfg = get_path_config()

        checkpoint_dir = path_cfg.get_expert_checkpoint_path(expert_name)

        if not checkpoint_dir.exists():
            return []

        # 获取所有子目录
        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir()]

        # 按修改时间排序
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return checkpoints

    except Exception as e:
        warnings.warn(f"列出checkpoint失败: {str(e)}")
        return []


# ===== 批量操作 =====

def scan_files(
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False
) -> List[Path]:
    """
    扫描目录下的文件

    Args:
        directory: 目录路径
        pattern: 文件匹配模式（如 "*.csv", "*.json"）
        recursive: 是否递归扫描子目录

    Returns:
        list: 文件路径列表

    Example:
        >>> scan_files('dataset', pattern='*.csv', recursive=True)
        [PosixPath('dataset/text/data.csv'), ...]
    """
    directory = Path(directory)
    validate_path_exists(directory, path_type='dir')

    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def batch_process_files(
        file_list: List[Path],
        process_fn: Callable[[Path], Any],
        error_handling: str = 'skip'
) -> List[Any]:
    """
    批量处理文件

    Args:
        file_list: 文件路径列表
        process_fn: 处理函数，接受Path参数
        error_handling: 错误处理方式 ('skip', 'raise', 'collect')

    Returns:
        list: 处理结果列表

    Example:
        >>> files = scan_files('dataset', '*.csv')
        >>> results = batch_process_files(files, load_csv)
    """
    results = []
    errors = []

    for file_path in file_list:
        try:
            result = process_fn(file_path)
            results.append(result)
        except Exception as e:
            if error_handling == 'raise':
                raise
            elif error_handling == 'skip':
                warnings.warn(f"处理文件失败 ({file_path}): {str(e)}")
                continue
            elif error_handling == 'collect':
                errors.append({'file': file_path, 'error': str(e)})
                continue

    if error_handling == 'collect' and errors:
        warnings.warn(f"批量处理完成，{len(errors)}个文件失败")
        results.append({'errors': errors})

    return results


# ===== 其他工具函数 =====

def get_file_size(filepath: Union[str, Path], human_readable: bool = True) -> Union[int, str]:
    """
    获取文件大小

    Args:
        filepath: 文件路径
        human_readable: 是否返回人类可读格式（如 "1.5 MB"）

    Returns:
        int 或 str: 文件大小（字节或可读格式）
    """
    filepath = Path(filepath)
    validate_path_exists(filepath, path_type='file')

    size_bytes = filepath.stat().st_size

    if not human_readable:
        return size_bytes

    # 转换为人类可读格式
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.2f} PB"


def copy_file_safe(
        src: Union[str, Path],
        dst: Union[str, Path],
        overwrite: bool = False
) -> Path:
    """
    安全复制文件

    Args:
        src: 源文件路径
        dst: 目标文件路径
        overwrite: 是否覆盖已存在的文件

    Returns:
        Path: 目标文件路径

    Raises:
        FileExistsError: 当目标文件已存在且overwrite=False时
    """
    src = Path(src)
    dst = Path(dst)

    validate_path_exists(src, path_type='file')
    ensure_dir(dst.parent)

    if dst.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在: {dst}")

    shutil.copy2(src, dst)
    return dst


def create_backup(
        filepath: Union[str, Path],
        backup_dir: Optional[Union[str, Path]] = None,
        timestamp: bool = True
) -> Path:
    """
    创建文件备份

    Args:
        filepath: 要备份的文件
        backup_dir: 备份目录（如果为None则在原目录创建）
        timestamp: 是否在备份文件名中添加时间戳

    Returns:
        Path: 备份文件路径
    """
    filepath = Path(filepath)
    validate_path_exists(filepath, path_type='file')

    # 确定备份目录
    if backup_dir is None:
        backup_dir = filepath.parent / 'backups'
    else:
        backup_dir = Path(backup_dir)

    ensure_dir(backup_dir)

    # 生成备份文件名
    if timestamp:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{filepath.stem}_{timestamp_str}{filepath.suffix}"
    else:
        backup_name = f"{filepath.stem}_backup{filepath.suffix}"

    backup_path = backup_dir / backup_name

    # 复制文件
    shutil.copy2(filepath, backup_path)

    return backup_path


# ===== 测试代码 =====
if __name__ == "__main__":
    print("=" * 60)
    print("文件工具测试")
    print("=" * 60)

    # 测试1：路径操作
    print("\n【测试1】路径操作")
    print("-" * 60)

    test_dir = ensure_dir('test_output/sub_dir')
    print(f"创建测试目录: {test_dir}")

    test_path = safe_path_join('test_output', 'sub_dir', 'test.txt')
    print(f"路径拼接结果: {test_path}")

    # 测试2：JSON操作
    print("\n【测试2】JSON操作")
    print("-" * 60)

    test_data = {
        'name': '测试数据',
        'value': 123,
        'items': ['a', 'b', 'c']
    }

    json_path = test_dir / 'test.json'
    save_json(test_data, json_path)
    print(f"保存JSON: {json_path}")

    loaded_data = load_json(json_path)
    print(f"加载JSON: {loaded_data}")

    # 更新JSON
    update_json(json_path, {'new_field': 'new_value'})
    updated_data = load_json(json_path)
    print(f"更新后的JSON: {updated_data}")

    # 测试3：CSV操作
    print("\n【测试3】CSV操作")
    print("-" * 60)

    csv_data = [
        {'name': 'Alice', 'age': 25, 'score': 90},
        {'name': 'Bob', 'age': 30, 'score': 85},
        {'name': 'Charlie', 'age': 28, 'score': 95}
    ]

    csv_path = test_dir / 'test.csv'
    save_csv(csv_data, csv_path)
    print(f"保存CSV: {csv_path}")

    loaded_csv = load_csv(csv_path)
    print(f"加载CSV: {len(loaded_csv)}行")
    print(f"第一行: {loaded_csv[0]}")

    # 测试4：文件扫描
    print("\n【测试4】文件扫描")
    print("-" * 60)

    json_files = scan_files('test_output', pattern='*.json', recursive=True)
    csv_files = scan_files('test_output', pattern='*.csv', recursive=True)

    print(f"找到JSON文件: {len(json_files)}个")
    print(f"找到CSV文件: {len(csv_files)}个")

    # 测试5：文件大小
    print("\n【测试5】文件大小")
    print("-" * 60)

    size = get_file_size(json_path, human_readable=True)
    print(f"JSON文件大小: {size}")

    # 测试6：文件备份
    print("\n【测试6】文件备份")
    print("-" * 60)

    backup_path = create_backup(json_path)
    print(f"创建备份: {backup_path}")

    # 清理测试文件
    print("\n清理测试文件...")
    shutil.rmtree('test_output')
    print("文件工具测试完成！")