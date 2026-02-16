"""
Text众包指令批次修复脚本
基于generate_instructions_text.py的稳定生成逻辑
核心特性:
1. 继承稳定的浏览器自动化功能
2. 批次完整性检查:如果批次中有任何ERROR,整个批次重新生成
3. 自动检测需要修复的批次范围
4. 精准检测"ERROR: 生成失败",支持多种引号格式
5. 三段式完整性检查(Definition/Emphasis/Things to Avoid)
6. 句号检查(可通过ENABLE_PERIOD_CHECK参数配置,默认关闭)
7. 详细错误报告,列出每条错误数据及具体问题
8. 使用与generate文件完全一致的硬编码Prompt
"""

import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re
from datetime import datetime
import chardet

# ==================== 配置参数 ====================
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DATASET_PATH = r"D:\MyPyProject\crowdsourcing_instruction_generator\dataset\Requirements_data\Text_data"
GPT_URL = "https://sass-node1.chatshare.biz/"

# 目标文件
CSV_FILE = "enhanced_CCHIT.csv"  # 可修改为需要修复的文件

# 批次参数
BATCH_SIZE = 10  # 批次大小(与首次生成保持一致)
CHECK_INTERVAL = 100
WAIT_NEW_RESPONSE_TIMEOUT = 60  # 等待新回复最多60秒
CONTENT_STABLE_CHECKS = 3  # 内容稳定性检查次数
ENABLE_PERIOD_CHECK = False  # 句号检测开关（默认关闭）

# ==================== 工具函数 ====================
class TextBatchRepairer:
    def __init__(self):
        self.driver = None
        self.current_tab = None
        self.repaired_count = 0
        self.error_log = []
        self.error_details = []  # 存储详细错误信息

        # 缓存成功的选择器
        self.cached_input_selector = None
        self.cached_button_selector = None

        # 记录发送前的回复数量
        self.response_count_before_send = 0

    def init_driver(self):
        """初始化Chrome浏览器"""
        print("\n" + "="*60)
        print("正在初始化浏览器...")
        print("="*60)

        if not os.path.exists(CHROME_PATH):
            raise FileNotFoundError(f"Chrome浏览器路径不存在: {CHROME_PATH}")
        print(f"✓ Chrome路径验证成功")

        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')

            user_data_dir = os.path.join(os.getcwd(), 'chrome_user_data')
            if not os.path.exists(user_data_dir):
                os.makedirs(user_data_dir)
            options.add_argument(f'--user-data-dir={user_data_dir}')

            options.add_argument('--disable-extensions')
            options.add_argument('--remote-debugging-port=9222')
            options.add_argument('--start-maximized')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option('excludeSwitches', ['enable-automation'])
            options.add_experimental_option('useAutomationExtension', False)

            print("✓ ChromeOptions配置完成")
            print("正在启动ChromeDriver...")
            self.driver = webdriver.Chrome(options=options)
            print(f"✓ ChromeDriver启动成功")

            print(f"\n正在导航到: {GPT_URL}")
            self.driver.get(GPT_URL)
            time.sleep(8)

            print(f"✓ 页面加载完成: {self.driver.title}")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\n✗ 浏览器初始化失败: {e}")
            raise

    def find_input_box(self, debug=False):
        """定位输入框 - 优化版,使用缓存"""
        if debug:
            print("🔍 定位输入框...")

        # 如果有缓存的选择器,优先使用
        if self.cached_input_selector:
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, self.cached_input_selector))
                )
                if element.is_displayed() and element.is_enabled():
                    if debug:
                        print(f"  ✓ 使用缓存选择器成功")
                    return element
                else:
                    self.cached_input_selector = None
            except:
                self.cached_input_selector = None

        selectors = [
            ("CSS", "div[contenteditable='true']"),
            ("CSS", "[contenteditable='true']"),
            ("CSS", "textarea"),
            ("CSS", "textarea[placeholder*='询问']"),
            ("CSS", "form textarea"),
            ("CSS", "div[class*='input'] textarea"),
        ]

        for selector_type, selector in selectors:
            try:
                if debug:
                    print(f"  尝试: {selector}")

                element = WebDriverWait(self.driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )

                if element.is_displayed() and element.is_enabled():
                    self.cached_input_selector = selector
                    if debug:
                        print(f"  ✓ 成功: {selector}")
                    return element

            except:
                continue

        raise NoSuchElementException("无法找到输入框")

    def find_submit_button(self):
        """定位提交按钮 - 优化版,使用缓存"""
        if self.cached_button_selector:
            try:
                button = self.driver.find_element(By.CSS_SELECTOR, self.cached_button_selector)
                if button.is_displayed() and button.is_enabled():
                    return button
                else:
                    self.cached_button_selector = None
            except:
                self.cached_button_selector = None

        selectors = [
            "button[data-testid='send-button']",
            "button[type='submit']",
            "button:has(svg)",
            "button[aria-label*='Send']",
            "button[aria-label*='发送']",
        ]

        for selector in selectors:
            try:
                buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for button in buttons:
                    if button.is_displayed() and button.is_enabled():
                        self.cached_button_selector = selector
                        return button
            except:
                continue

        return None

    def get_current_response_count(self):
        """
        基于实际DOM结构获取assistant回复数量
        使用data-message-author-role="assistant"作为准确标记
        """
        try:
            response_selectors = [
                "div[data-message-author-role='assistant']",
                "article[data-turn='assistant']",
                "article[data-testid*='conversation-turn'] div.markdown.prose",
            ]

            for selector in response_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)

                    if selector == "article[data-testid*='conversation-turn'] div.markdown.prose":
                        valid_count = 0
                        for elem in elements:
                            try:
                                parent = elem.find_element(By.XPATH, "./ancestor::article[contains(@data-testid, 'conversation-turn')]")
                                if parent:
                                    turn_attr = parent.get_attribute("data-turn")
                                    if turn_attr == "assistant":
                                        valid_count += 1
                            except:
                                continue
                        if valid_count > 0:
                            return valid_count
                    else:
                        if len(elements) > 0:
                            return len(elements)

                except:
                    continue

            print(f"  ⚠ 未找到assistant回复,返回0")
            return 0

        except Exception as e:
            print(f"  ⚠ 获取回复数量异常: {e}")
            return 0

    def wait_for_response(self, timeout=60):
        """
        等待新回复生成,支持长响应和瞬间生成
        返回: 新回复的文本 或 空字符串
        """
        print(f"\n⏳ 等待回复生成...")
        start_time = time.time()

        target_count = self.response_count_before_send + 1
        print(f"  期望回复数量: {target_count}")

        last_content = ""
        stable_count = 0

        while time.time() - start_time < timeout:
            try:
                current_count = self.get_current_response_count()

                if current_count >= target_count:
                    print(f"  ✓ 检测到新回复 (当前{current_count}条)")

                    response_selectors = [
                        "div[data-message-author-role='assistant']",
                        "article[data-turn='assistant']",
                    ]

                    for selector in response_selectors:
                        try:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            if len(elements) >= target_count:
                                response_elem = elements[target_count - 1]
                                current_content = response_elem.text

                                if current_content == last_content:
                                    stable_count += 1
                                    print(f"    内容稳定 ({stable_count}/{CONTENT_STABLE_CHECKS})", end='\r', flush=True)

                                    if stable_count >= CONTENT_STABLE_CHECKS:
                                        print(f"\n  ✓ 内容已稳定")
                                        return current_content
                                else:
                                    last_content = current_content
                                    stable_count = 0

                                time.sleep(1)
                                break
                        except:
                            continue

                time.sleep(2)

            except Exception as e:
                print(f"  ⚠ 检查异常: {e}")
                time.sleep(2)

        print(f"\n  ⚠ 等待超时 ({timeout}秒)")
        return ""

    def _clean_and_merge_content(self, text):
        """
        清理和合并内容：
        1. 移除换行符
        2. 处理bullet points，用分号分隔
        3. 清理多余空格
        4. 保留单独的"-"（表示无内容）
        """
        if not text:
            return ""

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line == '-':
                cleaned_lines.append(line)
            else:
                line = re.sub(r'^[•\-\*]\s+', '', line)
                line = line.strip()
                if line:
                    cleaned_lines.append(line)

        if len(cleaned_lines) > 1:
            result = '; '.join(cleaned_lines)
        elif len(cleaned_lines) == 1:
            result = cleaned_lines[0]
        else:
            result = ""

        if result != '-':
            result = re.sub(r'\s+', ' ', result).strip()

        return result

    def normalize_three_part_format(self, text):
        """
        标准化三段式格式：解析并合并多行内容
        """
        if not text:
            return text

        if 'Definition:' not in text or 'Emphasis & Caution:' not in text or 'Things to Avoid:' not in text:
            return text

        def_pos = text.find('Definition:')
        emp_pos = text.find('Emphasis & Caution:')
        avoid_pos = text.find('Things to Avoid:')

        if not (def_pos < emp_pos < avoid_pos):
            return text

        def_text = text[def_pos + len('Definition:'):emp_pos].strip()
        emp_text = text[emp_pos + len('Emphasis & Caution:'):avoid_pos].strip()
        avoid_text = text[avoid_pos + len('Things to Avoid:'):].strip()

        def_content = self._clean_and_merge_content(def_text)
        emp_content = self._clean_and_merge_content(emp_text)
        avoid_content = self._clean_and_merge_content(avoid_text)

        result = f"Definition: {def_content}\nEmphasis & Caution: {emp_content}\nThings to Avoid: {avoid_content}"
        return result

    def parse_text_instruction(self, response_text):
        """
        解析Text指令 - 三段式格式
        适配【需求N】格式和普通格式
        返回: instruction字符串 或 None
        """
        pattern = r'【需求\d+】\s*\n(.*?)(?=【需求\d+】|$)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        if len(matches) == 1:
            instruction = matches[0].strip()
            return self.normalize_three_part_format(instruction)

        parts = response_text.split('Definition:')
        for part in parts[1:]:
            if 'Emphasis & Caution:' in part and 'Things to Avoid:' in part:
                instruction = 'Definition:' + part.strip()
                return self.normalize_three_part_format(instruction)

        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]

        definition = None
        emphasis = None
        avoid = None

        for line in lines:
            if line.startswith('Definition:'):
                definition = line
            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                emphasis = line
            elif line.startswith('Things to Avoid:'):
                avoid = line

        if definition and emphasis and avoid:
            instruction = f"{definition}\n{emphasis}\n{avoid}"
            return self.normalize_three_part_format(instruction)
        else:
            return None

    def send_prompt(self, prompt_text, max_retries=3):
        """发送提示词到LLM"""
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print(f"\n📤 发送提示词...")
                    self.response_count_before_send = self.get_current_response_count()
                    print(f"  📊 当前页面已有 {self.response_count_before_send} 条回复")
                else:
                    print(f"  🔄 重试 {attempt}/{max_retries-1}...")

                input_box = self.find_input_box(debug=(attempt == 0))
                if not input_box:
                    if attempt < max_retries - 1:
                        self.driver.refresh()
                        time.sleep(8)
                        continue
                    return False

                self.driver.execute_script("arguments[0].focus();", input_box)
                time.sleep(0.3)

                tag_name = input_box.tag_name.lower()
                if tag_name == "textarea":
                    self.driver.execute_script("arguments[0].value = '';", input_box)
                else:
                    self.driver.execute_script("arguments[0].textContent = '';", input_box)

                time.sleep(0.3)

                if tag_name == "textarea":
                    self.driver.execute_script("""
                        var elem = arguments[0];
                        var text = arguments[1];
                        elem.value = text;
                        elem.dispatchEvent(new Event('input', { bubbles: true }));
                    """, input_box, prompt_text)
                else:
                    self.driver.execute_script("""
                        var elem = arguments[0];
                        var text = arguments[1];
                        elem.textContent = text;
                        elem.dispatchEvent(new Event('input', { bubbles: true }));
                        elem.focus();
                    """, input_box, prompt_text)

                time.sleep(1)

                current_value = input_box.get_attribute("value") if tag_name == "textarea" else input_box.text
                if not current_value or len(current_value) < 100:
                    if attempt < max_retries - 1:
                        continue
                    return False

                print(f"  ✓ 文本设置成功 ({len(current_value)} 字符)")

                button = self.find_submit_button()
                if button:
                    self.driver.execute_script("arguments[0].click();", button)
                    print(f"  ✓ 点击发送按钮")
                else:
                    input_box.send_keys(Keys.RETURN)
                    print(f"  ✓ 使用Enter发送")

                time.sleep(2)

                check_value = input_box.get_attribute("value") if tag_name == "textarea" else input_box.text
                if not check_value or len(check_value.strip()) < 50:
                    print("  ✓ 确认消息已发送")
                    return True

                time.sleep(2)
                return True

            except Exception as e:
                print(f"  ✗ 发送异常: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return False

        return False

    def process_batch(self, batch_data, start_idx):
        """
        处理单个批次的Text数据
        返回是否发生重试的标志
        """
        print(f"\n{'=' * 60}")
        print(f"处理批次 {start_idx + 1}-{start_idx + len(batch_data)}")
        print(f"{'=' * 60}")

        requirements_text = ""
        for i, (idx, requirement) in enumerate(batch_data):
            requirements_text += f"【需求{i+1}】\n{requirement}\n\n"

        prompt = f"""你是一个众包任务设计专家。请根据以下输入的需求文本,编写一个适合众包工人使用的英文任务指令。

核心原则:
1.极致精简:众包工人时间宝贵,请使用最简练的语言。
2.结构规范:严格按照下方定义的格式输出。
3.英语输出:无论输入是何种语言,输出必须是英文。

格式要求:
-Definition:使用简明扼要的祈使句描述主要目标。必须以 "In this task," 开头。
-Emphasis & Caution:仅指出极易出错或必须满足的特定条件。如无特别强调,填入 "-"。
-Things to Avoid:仅列出禁止的操作。如无特别避免事项,填入 "-"。

请为以下{len(batch_data)}条需求分别生成指令,严格按照以下格式输出:

{requirements_text}

输出格式:
【需求1】
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...

【需求2】
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ..."""

        retry_happened = False
        max_retries = 3

        for retry in range(max_retries):
            if retry > 0:
                print(f"\n⚠ 批次重试 {retry}/{max_retries-1}")
                retry_happened = True

            if not self.send_prompt(prompt):
                print(f"  ✗ 提示词发送失败")
                if retry < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(batch_data)}",
                        'error': f'提示词发送失败(重试{max_retries}次后失败)'
                    })
                    return [None] * len(batch_data), retry_happened

            response = self.wait_for_response(timeout=WAIT_NEW_RESPONSE_TIMEOUT)

            if not response or len(response) < 50:
                print(f"  ✗ 回复为空或过短")
                if retry < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(batch_data)}",
                        'error': f'生成错误(重试{max_retries}次后失败)'
                    })
                    return [None] * len(batch_data), retry_happened
            else:
                print(f"  ✓ 响应正常,准备解析")
                break

        instructions = []
        pattern = r'【需求\d+】\s*\n(.*?)(?=【需求\d+】|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        if len(matches) == len(batch_data):
            for match in matches:
                instruction = match.strip()
                instruction = self.normalize_three_part_format(instruction)
                instructions.append(instruction)
        else:
            parts = response.split('Definition:')
            for part in parts[1:]:
                if 'Emphasis & Caution:' in part and 'Things to Avoid:' in part:
                    instruction = 'Definition:' + part.strip()
                    instruction = self.normalize_three_part_format(instruction)
                    instructions.append(instruction)

        if len(instructions) != len(batch_data):
            print(f"  ⚠ 解析数量不匹配: 期望{len(batch_data)}, 实际{len(instructions)}")
            return [None] * len(batch_data), retry_happened

        return instructions, retry_happened

    def start_new_chat(self):
        """使用 Ctrl+Shift+O 快捷键开启新对话"""
        print("\n>>> 开启新对话...")
        try:
            from selenium.webdriver.common.action_chains import ActionChains

            print("  🔨 发送快捷键 Ctrl+Shift+O...")
            actions = ActionChains(self.driver)
            actions.key_down(Keys.CONTROL).key_down(Keys.SHIFT).send_keys('o').key_up(Keys.SHIFT).key_up(
                Keys.CONTROL).perform()
            print("  ✓ 快捷键已发送")

            time.sleep(3)

            self.cached_input_selector = None
            self.cached_button_selector = None
            self.response_count_before_send = 0

            print("  ✓ 新对话已就绪\n")

        except Exception as e:
            print(f"  ✗ 开启新对话失败: {e}")
            print("  ℹ 将继续在当前对话中处理")

    def validate_instruction_format(self, instruction):
        """验证指令格式完整性"""
        errors = []

        if not instruction or instruction.strip() == "":
            return False, ["指令为空"]

        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        if len(lines) < 3:
            return False, [f"行数不足(期望3行,实际{len(lines)}行)"]

        has_definition = False
        has_emphasis = False
        has_avoid = False

        for line in lines:
            if line.startswith('Definition:'):
                has_definition = True
                content = line[len('Definition:'):].strip()
                if not content.lower().startswith('in this task'):
                    errors.append("Definition未以'In this task'开头")
                if ENABLE_PERIOD_CHECK and not content.endswith('.'):
                    errors.append("Definition缺少结尾句号")

            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                has_emphasis = True
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if not content:
                    errors.append("Emphasis & Caution内容为空")
                if ENABLE_PERIOD_CHECK and content and content != '-' and not content.endswith('.'):
                    errors.append("Emphasis & Caution缺少结尾句号")

            elif line.startswith('Things to Avoid:'):
                has_avoid = True
                content = line[len('Things to Avoid:'):].strip()
                if content and content != '-' and not content.endswith('.'):
                    errors.append("Things to Avoid未以句号结尾（可能生成不完整）")

        if not has_definition:
            errors.append("缺少Definition部分")
        if not has_emphasis:
            errors.append("缺少Emphasis & Caution部分")
        if not has_avoid:
            errors.append("缺少Things to Avoid部分")

        is_valid = (has_definition and has_emphasis and has_avoid and len(errors) == 0)
        return is_valid, errors

    def detect_error_batches(self, df):
        """检测需要修复的批次"""
        print("\n" + "="*60)
        print("开始检测错误数据...")
        print("="*60)

        error_batches = []
        self.error_details = []

        total_rows = len(df)
        error_count = 0

        for i in range(0, total_rows, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_rows)
            batch_has_error = False
            batch_error_details = []

            for idx in range(i, batch_end):
                instruction = str(df.loc[idx, 'Instruction'])
                requirement_id = df.loc[idx, 'Requirement ID']
                row_num = idx + 1

                if 'ERROR' in instruction.upper() and '生成失败' in instruction:
                    batch_has_error = True
                    error_msg = "含有ERROR标记"
                    batch_error_details.append((row_num, requirement_id, error_msg))
                    continue

                if not instruction or instruction.strip() == '' or instruction == 'nan':
                    batch_has_error = True
                    error_msg = "指令为空"
                    batch_error_details.append((row_num, requirement_id, error_msg))
                    continue

                is_valid, format_errors = self.validate_instruction_format(instruction)
                if not is_valid:
                    batch_has_error = True
                    error_msg = "; ".join(format_errors)
                    batch_error_details.append((row_num, requirement_id, error_msg))

            if batch_has_error:
                error_count += len(batch_error_details)
                self.error_details.extend(batch_error_details)

                batch_data = []
                for idx in range(i, batch_end):
                    requirement_id = df.loc[idx, 'Requirement ID']
                    requirement_text = df.loc[idx, 'Requirement']
                    batch_data.append((requirement_id, requirement_text))

                error_batches.append((len(error_batches) + 1, i, batch_data))

        print(f"\n检测结果:")
        print(f"  总数据条数: {total_rows}")
        print(f"  错误数据条数: {error_count}")
        print(f"  需修复批次数: {len(error_batches)}")

        return error_batches

    def print_error_report(self):
        """打印详细错误报告"""
        if not self.error_details:
            return

        print("\n" + "="*60)
        print("详细错误报告")
        print("="*60)

        for row_num, req_id, error_msg in self.error_details:
            print(f"\n第{row_num}行: {req_id}")
            print(f"  错误: {error_msg}")

        print("\n" + "="*60)

    def repair_file(self, csv_path):
        """修复文件中的错误批次"""
        print(f"\n{'#' * 60}")
        print(f"# 处理文件: {os.path.basename(csv_path)}")
        print(f"{'#' * 60}")

        try:
            with open(csv_path, 'rb') as f:
                raw_data = f.read(100000)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                print(f"文件编码: {encoding}")

            try:
                df = pd.read_csv(csv_path, encoding=encoding)
            except:
                for enc in ['utf-8', 'gbk', 'gb18030', 'latin1']:
                    try:
                        df = pd.read_csv(csv_path, encoding=enc)
                        print(f"  ✓ 使用 {enc} 编码")
                        break
                    except:
                        continue
                else:
                    raise Exception("无法读取文件")

        except Exception as e:
            print(f"✗ 读取文件失败: {e}")
            return 0

        if 'Instruction' not in df.columns:
            df['Instruction'] = ''
        df['Instruction'] = df['Instruction'].astype(str)
        df.loc[df['Instruction'] == 'nan', 'Instruction'] = ''

        error_batches = self.detect_error_batches(df)

        if not error_batches:
            print("\n✓ 未发现需要修复的错误数据")
            self.print_error_report()
            return 0

        self.print_error_report()

        print(f"\n⚠️ 发现 {len(error_batches)} 个错误批次,共约 {len(error_batches) * BATCH_SIZE} 条数据需要修复")
        user_input = input("是否继续修复? (y/n): ").strip().lower()
        if user_input != 'y':
            print("❌ 用户取消修复")
            return 0

        self.start_new_chat()

        repaired_batches = 0
        for batch_num, start_idx, batch_data in error_batches:
            instructions, _ = self.process_batch(batch_data, start_idx)

            for i, instruction in enumerate(instructions):
                row_idx = start_idx + i
                if instruction:
                    instruction = self.normalize_three_part_format(instruction)
                    df.at[row_idx, 'Instruction'] = instruction
                    self.repaired_count += 1
                else:
                    df.at[row_idx, 'Instruction'] = "ERROR: 生成失败"

            repaired_batches += 1

            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 已保存进度: {start_idx + len(batch_data)}/{len(df)}")

            if repaired_batches < len(error_batches):
                self.start_new_chat()

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 文件修复完成: {os.path.basename(csv_path)}")
        print(f"  修复批次: {repaired_batches}")
        print(f"  修复数据: {self.repaired_count} 条\n")

        return self.repaired_count

    def run(self):
        """主运行函数"""
        start_time = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"{'Text批次完整性修复系统':^60}")
        print(f"{'=' * 60}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"批次大小: {BATCH_SIZE} 条/批")
        period_check_status = "启用" if ENABLE_PERIOD_CHECK else "关闭"
        print(f"检测功能: ERROR标记 + 三段式格式 + 句号检查({period_check_status})")
        print(f"目标文件: {CSV_FILE}")
        print(f"{'=' * 60}\n")

        try:
            self.init_driver()

            csv_path = os.path.join(DATASET_PATH, CSV_FILE)
            if os.path.exists(csv_path):
                total_repaired = self.repair_file(csv_path)
            else:
                print(f"✗ 文件不存在: {csv_path}")
                total_repaired = 0

            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n{'=' * 60}")
            print(f"{'修复完成':^60}")
            print(f"{'=' * 60}")
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"耗时: {duration}")
            print(f"总计修复: {total_repaired} 条数据")

            if self.error_log:
                print(f"\n错误日志:")
                for error in self.error_log:
                    print(f"  - {error['range']}: {error['error']}")

            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"\n✗ 运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.driver:
                input("按 Enter 关闭浏览器...")
                self.driver.quit()
                print("✓ 浏览器已关闭")


# ==================== 主程序 ====================
if __name__ == "__main__":
    repairer = TextBatchRepairer()
    repairer.run()