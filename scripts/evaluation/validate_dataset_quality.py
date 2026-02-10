"""
UML数据集质量验证脚本

验证UML数据集的质量和完整性，包括：
1. Instruction格式完整性（三段式结构）
2. Things to Avoid部分完整性
3. Description与Instruction的对应关系
4. 错误标记和空值检测
5. Description字段的JSON有效性
6. UML关键词密度检测
7. 内容长度合理性检测
8. 段落重复内容检测
9. Actors/Use Cases覆盖度检测

句号检测规则（启用--enable-period-check时）：
- Definition: 必须有实际内容且以句号结尾
- Emphasis & Caution: 若为"-"则无需句号，否则需要句号
- Things to Avoid: 若为"-"则无需句号，否则需要句号

作者：数据集验证系统
日期：2026-02-11
"""

import os
import sys
import json
import pandas as pd
import chardet
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any


class UMLDatasetValidator:
    """UML数据集质量验证器"""

    def __init__(self, dataset_path: str, enable_period_check: bool = False):
        """
        初始化验证器

        参数:
            dataset_path: CSV数据集文件路径
            enable_period_check: 是否检查句子结尾的句号
        """
        self.dataset_path = dataset_path
        self.enable_period_check = enable_period_check
        self.validation_results = []
        self.error_count = 0
        self.warning_count = 0

    def detect_encoding(self, filepath: str) -> str:
        """检测文件编码"""
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read(100000)
                result = chardet.detect(raw_data)
                return result['encoding']
        except Exception as e:
            print(f"检测编码错误: {e}")
            return 'utf-8'

    def load_dataset(self) -> pd.DataFrame:
        """加载数据集，自动检测编码"""
        print(f"\n加载数据集: {os.path.basename(self.dataset_path)}")

        encoding = self.detect_encoding(self.dataset_path)
        print(f"检测到编码: {encoding}")

        try:
            df = pd.read_csv(self.dataset_path, encoding=encoding)
            print(f"成功加载 {len(df)} 行数据\n")
            return df
        except Exception as e:
            print(f"使用 {encoding} 加载失败，尝试其他编码...")
            for enc in ['utf-8', 'gbk', 'gb18030', 'latin1']:
                try:
                    df = pd.read_csv(self.dataset_path, encoding=enc)
                    print(f"成功使用 {enc} 编码加载")
                    print(f"加载了 {len(df)} 行数据\n")
                    return df
                except:
                    continue
            raise Exception(f"加载数据集失败: {e}")

    def validate_json_description(self, description: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        验证Description字段的JSON结构

        返回:
            (是否有效, 错误消息列表)
        """
        errors = []

        if not description or pd.isna(description):
            errors.append("Description为空")
            return False, errors

        try:
            desc_json = json.loads(description)

            # 检查必需字段
            required_fields = ['actors', 'use_cases', 'relationships', 'overall_description']
            for field in required_fields:
                if field not in desc_json:
                    errors.append(f"缺少必需字段: {field}")

            # 验证actors结构
            if 'actors' in desc_json:
                if not isinstance(desc_json['actors'], list):
                    errors.append("'actors'应该是列表")
                else:
                    for idx, actor in enumerate(desc_json['actors']):
                        if not isinstance(actor, dict) or 'name' not in actor:
                            errors.append(f"actors索引{idx}结构无效")

            # 验证use_cases结构
            if 'use_cases' in desc_json:
                if not isinstance(desc_json['use_cases'], list):
                    errors.append("'use_cases'应该是列表")
                else:
                    for idx, uc in enumerate(desc_json['use_cases']):
                        if not isinstance(uc, dict):
                            errors.append(f"use_cases索引{idx}结构无效")
                        elif 'name' not in uc:
                            errors.append(f"use_cases索引{idx}缺少'name'字段")

            # 验证relationships结构
            if 'relationships' in desc_json:
                if not isinstance(desc_json['relationships'], list):
                    errors.append("'relationships'应该是列表")
                else:
                    for idx, rel in enumerate(desc_json['relationships']):
                        if not isinstance(rel, dict):
                            errors.append(f"relationships索引{idx}结构无效")
                        else:
                            required_rel_fields = ['type', 'from', 'to']
                            for field in required_rel_fields:
                                if field not in rel:
                                    errors.append(f"relationships索引{idx}缺少'{field}'字段")

        except json.JSONDecodeError as e:
            errors.append(f"JSON格式无效: {str(e)}")
            return False, errors
        except Exception as e:
            errors.append(f"未预期的错误: {str(e)}")
            return False, errors

        return len(errors) == 0, errors

    def validate_three_part_format(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        验证三段式指令格式:
        - Definition: ...
        - Emphasis & Caution: ...
        - Things to Avoid: ...

        返回:
            (是否有效, 错误消息列表)
        """
        errors = []

        if not instruction or pd.isna(instruction) or instruction.strip() == '':
            errors.append("Instruction为空")
            return False, errors

        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        if len(lines) < 3:
            errors.append(f"行数不足(期望3行，实际{len(lines)}行)")

        has_definition = False
        has_emphasis = False
        has_avoid = False

        for line in lines:
            if line.startswith('Definition:'):
                has_definition = True
                content = line[len('Definition:'):].strip()
                if not content.lower().startswith('in this task'):
                    errors.append("Definition未以'In this task'开头")
                if self.enable_period_check and not content.endswith('.'):
                    errors.append("Definition缺少结尾句号")

            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                has_emphasis = True
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if self.enable_period_check and content and content != '-' and not content.endswith('.'):
                    errors.append("Emphasis & Caution缺少结尾句号")

            elif line.startswith('Things to Avoid:'):
                has_avoid = True
                content = line[len('Things to Avoid:'):].strip()
                if self.enable_period_check and content and content != '-' and not content.endswith('.'):
                    errors.append("Things to Avoid缺少结尾句号")

        if not has_definition:
            errors.append("缺少Definition部分")
        if not has_emphasis:
            errors.append("缺少Emphasis & Caution部分")
        if not has_avoid:
            errors.append("缺少Things to Avoid部分")

        is_valid = (has_definition and has_emphasis and has_avoid and len(errors) == 0)
        return is_valid, errors

    def check_things_to_avoid_completeness(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        检查Things to Avoid部分是否完整，不只是复制回来的

        注意：根据项目规范，Things to Avoid可以是实际内容或显式"-"
        因此"-"是有效值，不应报警告

        返回:
            (是否完整, 警告消息列表)
        """
        warnings = []

        if not instruction or pd.isna(instruction):
            return False, ["Instruction为空"]

        # 提取Things to Avoid部分
        avoid_pattern = r'Things to Avoid:\s*(.+?)(?:\n|$)'
        match = re.search(avoid_pattern, instruction, re.DOTALL)

        if not match:
            warnings.append("无法找到Things to Avoid部分")
            return False, warnings

        avoid_content = match.group(1).strip()

        # 检查是否为显式的"-"（这是有效的，根据项目规范）
        if avoid_content == '-':
            return True, []  # "-"是有效值，不报警告

        # 检查是否为空
        if not avoid_content:
            warnings.append("Things to Avoid内容为空")
            return False, warnings

        # 检查是否为常见的不完整模式（但排除"-"）
        incomplete_patterns = [
            r'^TBD\s*$',
            r'^TODO\s*$',
            r'^N/A\s*$',
        ]

        for pattern in incomplete_patterns:
            if re.match(pattern, avoid_content, re.IGNORECASE):
                warnings.append(f"Things to Avoid看起来不完整: '{avoid_content}'")
                return False, warnings

        return True, []

    def validate_description_instruction_correspondence(
        self,
        description: str,
        instruction: str,
        row_num: int
    ) -> Tuple[bool, List[str]]:
        """
        验证Instruction是否与Description对应

        检查项：
        1. Description中的关键实体是否在Instruction中出现
        2. Use case名称是否被引用
        3. Actor名称是否被提及

        注意：这些检查可能产生误报，因为指令可能使用同义词或更通用的描述

        返回:
            (是否有效, 警告消息列表)
        """
        warnings = []

        if not description or not instruction:
            warnings.append("Description或Instruction为空")
            return False, warnings

        try:
            desc_json = json.loads(description)

            # 提取关键实体
            actors = [actor.get('name', '').lower() for actor in desc_json.get('actors', [])]
            use_cases = [uc.get('name', '').lower() for uc in desc_json.get('use_cases', [])]

            instruction_lower = instruction.lower()

            # 检查use cases是否被提及（宽松检查）
            use_cases_mentioned = sum(1 for uc in use_cases if uc and uc in instruction_lower)
            if len(use_cases) > 0 and use_cases_mentioned == 0:
                # 只在完全没有提及时才报警告
                warnings.append("Description中的use cases似乎没有在Instruction中提及")

            # 检查actors是否被提及（宽松检查）
            actors_mentioned = sum(1 for actor in actors if actor and actor in instruction_lower)
            if len(actors) > 0 and actors_mentioned == 0:
                # 只在完全没有提及时才报警告
                warnings.append("Description中的actors似乎没有在Instruction中提及")

            # 检查关系类型（可选检查，因为可能用不同词汇表达）
            relationships = desc_json.get('relationships', [])
            has_include = any(rel.get('type') == 'include' for rel in relationships)
            has_extend = any(rel.get('type') == 'extend' for rel in relationships)

            # 这些检查比较宽松，允许多种表达方式
            if has_include:
                include_keywords = ['include', 'required', 'mandatory', 'must', 'prerequisite']
                if not any(keyword in instruction_lower for keyword in include_keywords):
                    warnings.append("Description有'include'关系但Instruction中可能未体现")

            if has_extend:
                extend_keywords = ['extend', 'optional', 'conditional', 'may', 'can']
                if not any(keyword in instruction_lower for keyword in extend_keywords):
                    warnings.append("Description有'extend'关系但Instruction中可能未体现")

        except json.JSONDecodeError:
            warnings.append("无法解析Description的JSON进行对应关系检查")
            return False, warnings
        except Exception as e:
            warnings.append(f"检查对应关系时出错: {str(e)}")
            return False, warnings

        # 如果有警告，这是潜在问题但不一定无效（可能是误报）
        return len(warnings) == 0, warnings

    def check_error_markers(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        检查instruction中的ERROR标记

        返回:
            (是否干净, 错误消息列表)
        """
        errors = []

        if not instruction or pd.isna(instruction):
            errors.append("Instruction为空")
            return False, errors

        # 检查各种ERROR格式
        error_patterns = [
            r'ERROR\s*:',
            r'error\s*:',
            r'生成失败',
            r'generation failed',
            r'failed to generate',
        ]

        for pattern in error_patterns:
            if re.search(pattern, instruction, re.IGNORECASE):
                errors.append(f"包含ERROR标记: 匹配模式'{pattern}'")
                return False, errors

        return True, []

    def validate_keyword_density(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        检查UML关键术语密度

        验证Instruction是否包含足够的UML相关关键词

        返回:
            (是否合格, 警告消息列表)
        """
        warnings = []

        if not instruction or pd.isna(instruction):
            warnings.append("Instruction为空")
            return False, warnings

        instruction_lower = instruction.lower()

        # UML关键术语分类
        uml_keywords = {
            'relationships': ['include', 'extend', 'association', 'generalization', 'dependency'],
            'elements': ['actor', 'use case', 'usecase', 'use-case', 'system', 'boundary'],
            'qualifiers': ['required', 'optional', 'mandatory', 'conditional', 'prerequisite'],
            'workflow': ['workflow', 'process', 'interaction', 'execute', 'implement', 'trigger']
        }

        # 统计每类关键词的出现次数
        category_counts = {}
        total_keywords = 0

        for category, keywords in uml_keywords.items():
            count = sum(1 for keyword in keywords if keyword in instruction_lower)
            category_counts[category] = count
            total_keywords += count

        # 检查是否至少包含3个UML关键词
        if total_keywords < 3:
            warnings.append(f"UML关键术语过少(仅{total_keywords}个)，可能缺乏专业性")

        # 检查是否至少有2个类别有关键词
        categories_with_keywords = sum(1 for count in category_counts.values() if count > 0)
        if categories_with_keywords < 2:
            warnings.append(f"UML关键术语类别单一(仅{categories_with_keywords}类)，建议增加多样性")

        return len(warnings) == 0, warnings

    def validate_content_length(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        检查各段内容长度合理性

        验证三段式指令中每段内容的长度是否在合理范围内

        返回:
            (是否合格, 警告消息列表)
        """
        warnings = []

        if not instruction or pd.isna(instruction):
            warnings.append("Instruction为空")
            return False, warnings

        # 提取三段内容
        definition_match = re.search(r'Definition:\s*(.+?)(?=\n(?:Emphasis|$))', instruction, re.DOTALL)
        emphasis_match = re.search(r'Emphasis & Caution:\s*(.+?)(?=\nThings to Avoid:|$)', instruction, re.DOTALL)
        avoid_match = re.search(r'Things to Avoid:\s*(.+?)$', instruction, re.DOTALL)

        # 定义合理长度范围（字符数）
        length_ranges = {
            'Definition': (50, 500),
            'Emphasis & Caution': (10, 600),  # 允许"-"所以最小为10
            'Things to Avoid': (10, 400)      # 允许"-"所以最小为10
        }

        # 检查Definition长度
        if definition_match:
            definition_content = definition_match.group(1).strip()
            def_len = len(definition_content)
            min_len, max_len = length_ranges['Definition']

            if def_len < min_len:
                warnings.append(f"Definition过短({def_len}字符)，可能不完整")
            elif def_len > max_len:
                warnings.append(f"Definition过长({def_len}字符)，建议精简")

        # 检查Emphasis & Caution长度
        if emphasis_match:
            emphasis_content = emphasis_match.group(1).strip()
            if emphasis_content != '-':  # 不检查"-"的情况
                emp_len = len(emphasis_content)
                min_len, max_len = length_ranges['Emphasis & Caution']

                if emp_len < min_len:
                    warnings.append(f"Emphasis & Caution过短({emp_len}字符)，可能不完整")
                elif emp_len > max_len:
                    warnings.append(f"Emphasis & Caution过长({emp_len}字符)，建议精简")

        # 检查Things to Avoid长度
        if avoid_match:
            avoid_content = avoid_match.group(1).strip()
            if avoid_content != '-':  # 不检查"-"的情况
                avoid_len = len(avoid_content)
                min_len, max_len = length_ranges['Things to Avoid']

                if avoid_len < min_len:
                    warnings.append(f"Things to Avoid过短({avoid_len}字符)，可能不完整")
                elif avoid_len > max_len:
                    warnings.append(f"Things to Avoid过长({avoid_len}字符)，建议精简")

        return len(warnings) == 0, warnings

    def validate_content_duplication(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        检查段落间重复内容

        验证三段式指令的各段之间是否存在过多重复内容

        返回:
            (是否合格, 警告消息列表)
        """
        warnings = []

        if not instruction or pd.isna(instruction):
            warnings.append("Instruction为空")
            return False, warnings

        # 提取三段内容
        definition_match = re.search(r'Definition:\s*(.+?)(?=\n(?:Emphasis|$))', instruction, re.DOTALL)
        emphasis_match = re.search(r'Emphasis & Caution:\s*(.+?)(?=\nThings to Avoid:|$)', instruction, re.DOTALL)
        avoid_match = re.search(r'Things to Avoid:\s*(.+?)$', instruction, re.DOTALL)

        if not (definition_match and emphasis_match and avoid_match):
            # 如果无法提取，不检查重复
            return True, []

        definition_content = definition_match.group(1).strip().lower()
        emphasis_content = emphasis_match.group(1).strip().lower()
        avoid_content = avoid_match.group(1).strip().lower()

        # 跳过"-"的检查
        if emphasis_content == '-' or avoid_content == '-':
            return True, []

        # 计算段落间的重复词汇比例
        def get_words(text):
            # 提取单词（长度>=4的英文单词）
            words = re.findall(r'\b[a-z]{4,}\b', text)
            return set(words)

        def_words = get_words(definition_content)
        emp_words = get_words(emphasis_content)
        avoid_words = get_words(avoid_content)

        # 检查Definition和Emphasis的重复
        if len(def_words) > 0 and len(emp_words) > 0:
            overlap_def_emp = len(def_words & emp_words)
            overlap_ratio = overlap_def_emp / min(len(def_words), len(emp_words))

            if overlap_ratio > 0.7:
                warnings.append(f"Definition和Emphasis & Caution内容重复度过高({overlap_ratio:.1%})")

        # 检查Definition和Things to Avoid的重复
        if len(def_words) > 0 and len(avoid_words) > 0:
            overlap_def_avoid = len(def_words & avoid_words)
            overlap_ratio = overlap_def_avoid / min(len(def_words), len(avoid_words))

            if overlap_ratio > 0.7:
                warnings.append(f"Definition和Things to Avoid内容重复度过高({overlap_ratio:.1%})")

        # 检查Emphasis和Things to Avoid的重复
        if len(emp_words) > 0 and len(avoid_words) > 0:
            overlap_emp_avoid = len(emp_words & avoid_words)
            overlap_ratio = overlap_emp_avoid / min(len(emp_words), len(avoid_words))

            if overlap_ratio > 0.7:
                warnings.append(f"Emphasis & Caution和Things to Avoid内容重复度过高({overlap_ratio:.1%})")

        return len(warnings) == 0, warnings

    def validate_coverage(self, description: str, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """
        增强版覆盖度检测

        检查Description中的actors和use_cases是否都在Instruction中被充分提及

        返回:
            (是否合格, 警告消息列表)
        """
        warnings = []

        if not description or not instruction:
            warnings.append("Description或Instruction为空")
            return False, warnings

        try:
            desc_json = json.loads(description)

            actors = desc_json.get('actors', [])
            use_cases = desc_json.get('use_cases', [])

            instruction_lower = instruction.lower()

            # 统计actors覆盖率
            if actors:
                actor_names = [actor.get('name', '').lower() for actor in actors if actor.get('name')]
                mentioned_actors = [name for name in actor_names if name and name in instruction_lower]
                coverage_rate = len(mentioned_actors) / len(actor_names) if actor_names else 0

                if coverage_rate < 0.5:
                    warnings.append(f"Actors覆盖率过低({coverage_rate:.0%})，仅提及{len(mentioned_actors)}/{len(actor_names)}个")

            # 统计use_cases覆盖率
            if use_cases:
                uc_names = [uc.get('name', '').lower() for uc in use_cases if uc.get('name')]
                mentioned_ucs = [name for name in uc_names if name and name in instruction_lower]
                coverage_rate = len(mentioned_ucs) / len(uc_names) if uc_names else 0

                if coverage_rate < 0.5:
                    warnings.append(f"Use Cases覆盖率过低({coverage_rate:.0%})，仅提及{len(mentioned_ucs)}/{len(uc_names)}个")

        except json.JSONDecodeError:
            warnings.append("无法解析Description的JSON进行覆盖度检查")
            return False, warnings
        except Exception as e:
            warnings.append(f"检查覆盖度时出错: {str(e)}")
            return False, warnings

        return len(warnings) == 0, warnings

    def validate_row(self, row: pd.Series, row_num: int) -> Dict[str, Any]:
        """
        验证单行数据

        返回:
            包含验证结果的字典
        """
        result = {
            'row_num': row_num,
            'header': row.get('Header', 'N/A'),
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        description = str(row.get('Description', ''))
        instruction = str(row.get('Instruction', ''))

        # 检查1: JSON Description有效性
        json_valid, json_errors = self.validate_json_description(description, row_num)
        if not json_valid:
            result['is_valid'] = False
            result['errors'].extend([f"[JSON] {err}" for err in json_errors])

        # 检查2: Error标记
        error_clean, error_messages = self.check_error_markers(instruction, row_num)
        if not error_clean:
            result['is_valid'] = False
            result['errors'].extend([f"[ERROR] {err}" for err in error_messages])

        # 检查3: 三段式格式
        format_valid, format_errors = self.validate_three_part_format(instruction, row_num)
        if not format_valid:
            result['is_valid'] = False
            result['errors'].extend([f"[FORMAT] {err}" for err in format_errors])

        # 检查4: Things to Avoid完整性（修改后"-"是有效值）
        avoid_complete, avoid_warnings = self.check_things_to_avoid_completeness(instruction, row_num)
        if not avoid_complete:
            result['warnings'].extend([f"[AVOID] {warn}" for warn in avoid_warnings])

        # 检查5: Description-Instruction对应关系（可选检查，可能误报）
        corr_valid, corr_warnings = self.validate_description_instruction_correspondence(
            description, instruction, row_num
        )
        if not corr_valid:
            result['warnings'].extend([f"[CORRESPONDENCE] {warn}" for warn in corr_warnings])

        # 检查6: UML关键词密度
        keyword_valid, keyword_warnings = self.validate_keyword_density(instruction, row_num)
        if not keyword_valid:
            result['warnings'].extend([f"[KEYWORD] {warn}" for warn in keyword_warnings])

        # 检查7: 内容长度合理性
        length_valid, length_warnings = self.validate_content_length(instruction, row_num)
        if not length_valid:
            result['warnings'].extend([f"[LENGTH] {warn}" for warn in length_warnings])

        # 检查8: 段落重复内容
        dup_valid, dup_warnings = self.validate_content_duplication(instruction, row_num)
        if not dup_valid:
            result['warnings'].extend([f"[DUPLICATION] {warn}" for warn in dup_warnings])

        # 检查9: 覆盖度检测
        cov_valid, cov_warnings = self.validate_coverage(description, instruction, row_num)
        if not cov_valid:
            result['warnings'].extend([f"[COVERAGE] {warn}" for warn in cov_warnings])

        return result

    def validate_dataset(self) -> List[Dict[str, Any]]:
        """
        验证整个数据集

        返回:
            每行验证结果的列表
        """
        print("=" * 80)
        print("UML数据集质量验证".center(80))
        print("=" * 80)
        print(f"数据集: {os.path.basename(self.dataset_path)}")
        print(f"句号检查: {'启用' if self.enable_period_check else '禁用'}")
        print("=" * 80)
        print()

        df = self.load_dataset()

        print("开始验证...\n")

        results = []
        for idx, row in df.iterrows():
            row_num = idx + 1
            if row_num % 100 == 0:
                print(f"进度: {row_num}/{len(df)} 行已验证")

            result = self.validate_row(row, row_num)
            results.append(result)

            if not result['is_valid']:
                self.error_count += 1
            if result['warnings']:
                self.warning_count += 1

        self.validation_results = results
        return results

    def generate_report(self, save_path: str = None) -> str:
        """
        生成详细验证报告

        参数:
            save_path: 可选的报告保存路径

        返回:
            报告摘要字符串
        """
        if not self.validation_results:
            return "无验证结果。请先运行validate_dataset()。"

        print("\n" + "=" * 80)
        print("验证报告".center(80))
        print("=" * 80)

        total_rows = len(self.validation_results)
        valid_rows = sum(1 for r in self.validation_results if r['is_valid'])
        invalid_rows = total_rows - valid_rows
        rows_with_warnings = sum(1 for r in self.validation_results if r['warnings'])

        summary = f"""
总行数: {total_rows}
有效行数: {valid_rows} ({valid_rows/total_rows*100:.1f}%)
无效行数: {invalid_rows} ({invalid_rows/total_rows*100:.1f}%)
有警告的行数: {rows_with_warnings} ({rows_with_warnings/total_rows*100:.1f}%)
"""

        print(summary)

        # 打印详细错误
        if invalid_rows > 0:
            print("\n" + "-" * 80)
            print("详细错误:")
            print("-" * 80)

            for result in self.validation_results:
                if not result['is_valid']:
                    print(f"\n第 {result['row_num']} 行 [{result['header'][:40]}...]:")
                    for error in result['errors']:
                        print(f"  错误: {error}")
                    for warning in result['warnings']:
                        print(f"  警告: {warning}")

        # 打印警告（只打印有效行的警告）
        if rows_with_warnings > 0:
            print("\n" + "-" * 80)
            print("详细警告:")
            print("-" * 80)

            warning_count = 0
            for result in self.validation_results:
                if result['warnings'] and result['is_valid']:  # 只显示有效行的警告
                    warning_count += 1
                    if warning_count <= 20:  # 限制输出
                        print(f"\n第 {result['row_num']} 行 [{result['header'][:40]}...]:")
                        for warning in result['warnings']:
                            print(f"  警告: {warning}")

            if warning_count > 20:
                print(f"\n... 还有 {warning_count - 20} 行有警告")

        # 保存到CSV
        if save_path:
            self.save_report_csv(save_path)
            print(f"\n详细报告已保存至: {save_path}")

        print("=" * 80)

        return summary

    def save_report_csv(self, save_path: str):
        """保存验证报告到CSV文件"""
        report_data = []

        for result in self.validation_results:
            report_data.append({
                '行号': result['row_num'],
                'Header': result['header'],
                '是否有效': result['is_valid'],
                '错误': ' | '.join(result['errors']),
                '警告': ' | '.join(result['warnings'])
            })

        df_report = pd.DataFrame(report_data)
        df_report.to_csv(save_path, index=False, encoding='utf-8-sig')

    def save_problematic_instructions(self, output_dir: str, df: pd.DataFrame):
        """
        保存有问题的指令到单独文件，供进一步分析

        参数:
            output_dir: 输出目录
            df: 原始数据集DataFrame
        """
        os.makedirs(output_dir, exist_ok=True)

        # 收集所有有问题的行（有错误或警告）
        problematic_rows = []
        for result in self.validation_results:
            if not result['is_valid'] or result['warnings']:
                row_num = result['row_num']
                row_data = df.iloc[row_num - 1]  # 转换为0-based索引

                problematic_rows.append({
                    '行号': result['row_num'],
                    'Header': result['header'],
                    'Description': row_data.get('Description', ''),
                    'Instruction': row_data.get('Instruction', ''),
                    '错误': ' | '.join(result['errors']) if result['errors'] else '',
                    '警告': ' | '.join(result['warnings']) if result['warnings'] else '',
                    '是否有效': result['is_valid']
                })

        if problematic_rows:
            # 保存为CSV供LLM分析
            df_problematic = pd.DataFrame(problematic_rows)
            output_path = os.path.join(output_dir, 'problematic_instructions_for_llm_review.csv')
            df_problematic.to_csv(output_path, index=False, encoding='utf-8-sig')

            print(f"\n有问题的指令已保存至: {output_path}")
            print(f"共 {len(problematic_rows)} 条需要人工或LLM审查")

            # 生成简要统计
            error_only = sum(1 for r in problematic_rows if r['错误'] and not r['警告'])
            warning_only = sum(1 for r in problematic_rows if r['警告'] and not r['错误'])
            both = sum(1 for r in problematic_rows if r['错误'] and r['警告'])

            print(f"  - 仅有错误: {error_only} 条")
            print(f"  - 仅有警告: {warning_only} 条")
            print(f"  - 同时有错误和警告: {both} 条")
        else:
            print("\n没有发现问题指令")

    def get_error_rows(self) -> List[int]:
        """获取有错误的行号列表"""
        return [r['row_num'] for r in self.validation_results if not r['is_valid']]

    def get_warning_rows(self) -> List[int]:
        """获取有警告的行号列表"""
        return [r['row_num'] for r in self.validation_results if r['warnings']]


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='验证UML数据集质量')
    parser.add_argument('--dataset', type=str,
                       default='dataset/uml/uml_dataset_qwen3_v3.csv',
                       help='数据集CSV文件路径')
    parser.add_argument('--enable-period-check', action='store_true',
                       help='启用句子结尾句号检查')
    parser.add_argument('--report-output', type=str,
                       default=None,
                       help='验证报告CSV保存路径')

    args = parser.parse_args()

    # 初始化验证器
    validator = UMLDatasetValidator(
        dataset_path=args.dataset,
        enable_period_check=args.enable_period_check
    )

    # 运行验证
    start_time = datetime.now()
    results = validator.validate_dataset()
    end_time = datetime.now()

    # 加载原始数据集用于保存问题指令
    df = validator.load_dataset()

    # 生成报告
    if args.report_output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.report_output = f'outputs/validation/uml_validation_report_{timestamp}.csv'

    os.makedirs(os.path.dirname(args.report_output), exist_ok=True)

    summary = validator.generate_report(save_path=args.report_output)

    # 保存有问题的指令到单独文件供LLM分析
    problematic_output_dir = os.path.join(os.path.dirname(args.report_output), 'problematic_instructions')
    validator.save_problematic_instructions(problematic_output_dir, df)

    # 打印执行时间
    duration = end_time - start_time
    print(f"\n验证完成，耗时: {duration}")

    # 结果汇总
    error_count = validator.error_count
    warning_count = validator.warning_count

    print(f"\n{'=' * 80}")
    print(f"验证汇总".center(80))
    print(f"{'=' * 80}")
    print(f"错误总数: {error_count}")
    print(f"警告总数: {warning_count}")
    print(f"验证报告: {args.report_output}")
    if error_count > 0 or warning_count > 0:
        print(f"问题指令: {problematic_output_dir}/problematic_instructions_for_llm_review.csv")
    print(f"{'=' * 80}")

    if error_count > 0:
        print(f"\n错误行号: {validator.get_error_rows()[:20]}")
        if len(validator.get_error_rows()) > 20:
            print(f"... 还有 {len(validator.get_error_rows()) - 20} 行有错误")
        print(f"\n修复错误请运行:")
        print(f"python scripts/dataset_preparation/uml_dataset_regenerate.py")

    if warning_count > 0:
        print(f"\n警告行号: {validator.get_warning_rows()[:20]}")
        if len(validator.get_warning_rows()) > 20:
            print(f"... 还有 {len(validator.get_warning_rows()) - 20} 行有警告")
        print(f"\n警告可能是误报，建议人工或LLM审查问题指令文件")

    if error_count == 0 and warning_count == 0:
        print("\n数据集质量优秀！未发现问题。")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())