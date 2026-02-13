"""
众包指令批次修复脚本 - 完整版 (改进错误检测)
基于稳定的generate.py代码
核心特性:
1. 继承generate.py的所有稳定功能
2. 批次完整性检查:如果批次中有任何ERROR,整个批次重新生成
3. 自动检测需要修复的批次范围
4. ✨ 新增:精准检测"ERROR: 生成失败",支持多种引号格式
5. ✨ 新增:详细错误报告,列出每条错误数据
"""

import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re
from datetime import datetime
import chardet

# ==================== 配置参数 ====================
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DATASET_PATH = r"D:\MyPyProject\crowdsourcing_instruction_generator\dataset\Requirements_data\Text_data"
GPT_URL = "https://sass-node1.chatshare.biz/"

BATCH_SIZE = 10  # 批次大小

# 提示词模板
SYSTEM_PROMPT = """你是一个众包任务设计专家。请根据以下输入的需求文本,编写一个适合众包工人使用的英文任务指令。

核心原则:
1.极致精简:众包工人时间宝贵,请使用最简练的语言。
2.结构规范:严格按照下方定义的格式输出。
3.英语输出无论输入是何种语言,输出必须是英文。

格式要求:
-Definition:使用简明扼要的祈使句描述主要目标。必须以 "In this task," 开头。
-Emphasis & Caution:仅指出极易出错或必须满足的特定条件。如无特别强调,填入 "-"。
-Things to Avoid:仅列出禁止的操作。如无特别避免事项,填入 "-"。

请为以下{count}条需求分别生成指令,严格按照以下格式输出:

{requirements}

请严格按照以下格式输出每条指令,不要添加额外说明:

【需求1】
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...

【需求2】
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...

(依此类推)
"""


# ==================== 修复工具类 ====================
class BatchRepairer:
    """批次修复器 - 继承generate.py的稳定功能"""

    def __init__(self):
        self.driver = None
        self.repaired_count = 0
        self.error_log = []
        self.error_details = []  # ✨ 新增:存储详细错误信息

        # 【从generate继承】缓存成功的选择器
        self.cached_input_selector = None
        self.cached_button_selector = None

        # 【从generate继承】记录发送前的回复数量
        self.response_count_before_send = 0

    def init_driver(self):
        """【从generate.py完整复制】初始化Chrome浏览器"""
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

            user_data_dir = os.path.join(os.getcwd(), 'chrome_user_data_repair')
            if not os.path.exists(user_data_dir):
                os.makedirs(user_data_dir)
            options.add_argument(f'--user-data-dir={user_data_dir}')

            options.add_argument('--disable-extensions')
            options.add_argument('--remote-debugging-port=9223')
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
        """【从generate.py完整复制】定位输入框 - 优化版,使用缓存"""
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

        # 调整选择器优先级
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
        """【从generate.py完整复制】定位提交按钮 - 优化版,使用缓存"""
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
        """【从generate.py完整复制】获取当前页面的回复数量"""
        try:
            response_selectors = [
                "div[class*='markdown']",
                "div[data-message-author-role='assistant']",
                "div[class*='message']",
                "[class*='assistant']"
            ]

            for selector in response_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements and len(elements) > 0:
                        return len(elements)
                except:
                    continue
            return 0
        except:
            return 0

    def check_response_still_updating(self):
        """【从generate.py完整复制】核心检测方法:通过内容变化判断是否还在生成"""
        try:
            response_selectors = [
                "div[class*='markdown']",
                "div[data-message-author-role='assistant']",
                "div[class*='message']",
                "[class*='assistant']"
            ]

            for selector in response_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    current_count = len(elements)

                    if current_count <= self.response_count_before_send:
                        return True

                    if elements and current_count > 0:
                        new_response = elements[-1]
                        first_text = new_response.text
                        first_len = len(first_text)

                        time.sleep(0.8)

                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if len(elements) > 0:
                            new_response = elements[-1]
                            second_text = new_response.text
                            second_len = len(second_text)

                            if second_len > first_len:
                                return True
                            return False
                except:
                    continue

            return False
        except:
            return False

    def wait_for_response_complete(self, timeout=300):
        """【从generate.py完整复制】优化版:只使用内容更新检测"""
        print("  等待生成...", end='', flush=True)
        start_time = time.time()
        last_dot_time = start_time

        # 等待新回复出现
        print(" [等待响应]", end='', flush=True)
        response_appeared = False

        for _ in range(20):
            try:
                current_count = self.get_current_response_count()
                if current_count > self.response_count_before_send:
                    response_appeared = True
                    print(" ✓", end='', flush=True)
                    break
            except:
                pass
            time.sleep(0.5)

        if not response_appeared:
            print(" [未检测到新响应,继续等待]", end='', flush=True)

        # 监控内容更新,直到稳定
        stable_count = 0
        required_stable_checks = 3

        while time.time() - start_time < timeout:
            try:
                is_updating = self.check_response_still_updating()

                if is_updating:
                    stable_count = 0
                    print(".", end='', flush=True)
                else:
                    stable_count += 1

                    if stable_count >= required_stable_checks:
                        print(" ✓ 完成")
                        return True
                    else:
                        print(".", end='', flush=True)

                current_time = time.time()
                if current_time - last_dot_time >= 5:
                    elapsed = int(current_time - start_time)
                    print(f" [{elapsed}s]", end='', flush=True)
                    last_dot_time = current_time

                time.sleep(0.5)

            except Exception as e:
                print(f" ⚠ ", end='', flush=True)
                time.sleep(1)

        print(" ✗ 超时")
        return False

    def extract_response(self):
        """【从generate.py完整复制】提取LLM回复内容"""
        try:
            response_selectors = [
                "div[class*='markdown']",
                "div[data-message-author-role='assistant']",
                "div[class*='message']",
                "[class*='assistant']"
            ]

            for selector in response_selectors:
                try:
                    response_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    current_count = len(response_elements)

                    if current_count > self.response_count_before_send:
                        last_response = response_elements[-1].text
                        if last_response and len(last_response) > 10:
                            print(f"  ✓ 提取到回复 ({len(last_response)} 字符)")
                            return last_response
                except:
                    continue

            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            print(f"  ⚠ 使用body文本")
            return body_text

        except Exception as e:
            print(f"✗ 提取回复失败: {e}")
            return ""

    def parse_instructions(self, response_text, expected_count):
        """【从generate.py完整复制】解析LLM回复,提取指令"""
        instructions = []

        pattern = r'【需求\d+】\s*\n(.*?)(?=【需求\d+】|$)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        if len(matches) == expected_count:
            for match in matches:
                instructions.append(match.strip())
        else:
            parts = response_text.split('Definition:')
            for part in parts[1:]:
                if 'Emphasis & Caution:' in part and 'Things to Avoid:' in part:
                    instructions.append('Definition:' + part.strip())

        return instructions

    def send_prompt(self, prompt_text, max_retries=3):
        """【从generate.py完整复制】发送提示词到LLM - 优化版"""
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

    def start_new_chat(self):
        """【从generate.py完整复制】使用 Ctrl+Shift+O 快捷键开启新对话"""
        print("\n>>> 开启新对话...")
        try:
            print("  🔨 发送快捷键 Ctrl+Shift+O...")
            actions = ActionChains(self.driver)
            actions.key_down(Keys.CONTROL).key_down(Keys.SHIFT).send_keys('o').key_up(Keys.SHIFT).key_up(
                Keys.CONTROL).perform()
            print("  ✓ 快捷键已发送")

            time.sleep(3)

            self.cached_input_selector = None
            self.cached_button_selector = None
            self.response_count_before_send = 0

            print("  ✓ 新对话已就绪")

        except Exception as e:
            print(f"  ✗ 开启新对话失败: {e}")
            print("  ℹ 将继续在当前对话中处理")

    def check_instruction_has_error(self, instruction):
        """
        ✨ 新增方法:检查指令是否包含错误标志
        支持多种引号格式的"ERROR: 生成失败"

        Returns:
            (is_error: bool, error_type: str)
        """
        instruction_str = str(instruction).strip()

        # 定义要检测的错误模式(支持多种引号格式)
        error_patterns = [
            'ERROR: 生成失败',      # 标准格式
            'ERROR:生成失败',       # 无空格版本
            '"ERROR: 生成失败"',    # 英文双引号
            "'ERROR: 生成失败'",    # 英文单引号
            '"ERROR: 生成失败"',    # 中文双引号
            # 中文单引号
            '「ERROR: 生成失败」',  # 日文方括号
            '『ERROR: 生成失败』',  # 日文书名号
            '【ERROR: 生成失败】',  # 中文方括号
        ]

        # 检查是否匹配任何错误模式
        for pattern in error_patterns:
            if pattern in instruction_str:
                return True, f"包含错误标志: {pattern}"

        # 检查空值
        if pd.isna(instruction) or instruction_str == '' or instruction_str == 'nan':
            return True, "空值"

        # 检查内容过短(可能生成失败)
        if len(instruction_str) < 20:
            return True, f"内容过短({len(instruction_str)}字符)"

        return False, None

    def detect_error_batches(self, df):
        """
        ✨ 改进版:检测需要修复的批次
        - 专门检测"ERROR: 生成失败"及其变体
        - 支持多种引号格式
        - 生成详细错误报告

        返回: [(批次编号, 起始索引, 需求列表), ...]
        """
        print("\n" + "="*60)
        print("🔍 扫描错误批次...")
        print("="*60)

        total_rows = len(df)
        error_batches = []
        self.error_details = []  # 清空之前的错误详情

        # 按批次扫描
        for batch_start in range(0, total_rows, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_rows)
            batch_df = df.iloc[batch_start:batch_end]

            # 检查这个批次中是否有错误
            has_error = False
            batch_errors = []

            for idx, row in batch_df.iterrows():
                instruction = row.get('Instruction', '')
                requirement = str(row.get('Low_Requirements', ''))

                # 使用新的错误检测方法
                is_error, error_type = self.check_instruction_has_error(instruction)

                if is_error:
                    has_error = True

                    # 记录详细错误信息
                    error_info = {
                        'row_number': idx + 1,  # Excel中的行号(从1开始,包含表头)
                        'csv_index': idx,  # DataFrame中的索引
                        'batch_number': len(error_batches) + 1,
                        'batch_range': f"{batch_start + 1}-{batch_end}",
                        'requirement': requirement[:60] + '...' if len(requirement) > 60 else requirement,
                        'instruction': str(instruction)[:80] + '...' if len(str(instruction)) > 80 else str(instruction),
                        'error_type': error_type,
                        'instruction_length': len(str(instruction))
                    }
                    batch_errors.append(error_info)
                    self.error_details.append(error_info)

            # 如果批次中有错误,记录整个批次
            if has_error:
                batch_requirements = batch_df['Low_Requirements'].tolist()
                batch_num = len(error_batches) + 1
                error_batches.append((batch_num, batch_start, batch_requirements))

                print(f"\n  ⚠️ 批次 {batch_start + 1}-{batch_end}: 发现 {len(batch_errors)}/{len(batch_df)} 条错误")

                # 显示本批次的错误详情(最多显示3条)
                for i, err in enumerate(batch_errors[:3]):
                    print(f"     • 行{err['row_number']}: {err['error_type']}")
                if len(batch_errors) > 3:
                    print(f"     ... 还有 {len(batch_errors) - 3} 条错误")

        # 打印总体统计
        print(f"\n{'='*60}")
        print(f"📊 扫描统计:")
        print(f"  总数据行数: {total_rows}")
        print(f"  总批次数: {(total_rows + BATCH_SIZE - 1) // BATCH_SIZE}")
        print(f"  错误批次数: {len(error_batches)}")
        print(f"  错误数据条数: {len(self.error_details)}")
        print(f"  需重新生成约: {len(error_batches) * BATCH_SIZE} 条")
        print("="*60)

        return error_batches

    def print_error_report(self):
        """✨ 新增方法:打印详细的错误数据报告"""
        if not self.error_details:
            print("\n✓ 未发现错误数据")
            return

        print("\n" + "="*80)
        print(f"{'📋 错误数据详细报告':^80}")
        print("="*80)

        # 按错误类型分组统计
        error_by_type = {}
        for err in self.error_details:
            err_type = err['error_type']
            if err_type not in error_by_type:
                error_by_type[err_type] = []
            error_by_type[err_type].append(err)

        # 打印错误类型统计
        print(f"\n📊 错误类型统计:")
        for err_type, errors in sorted(error_by_type.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  • {err_type}: {len(errors)} 条")

        # 打印详细错误列表
        print(f"\n📝 详细错误列表:")
        print(f"{'行号':<8} {'批次范围':<15} {'错误类型':<30} {'需求内容预览':<40}")
        print("-" * 95)

        for err in self.error_details:
            row_num_str = f"{err['row_number']}"
            batch_str = f"{err['batch_range']}"
            error_type_str = f"{err['error_type'][:28]}"
            req_str = f"{err['requirement'][:38]}"

            print(f"{row_num_str:<8} {batch_str:<15} {error_type_str:<30} {req_str:<40}")

        print("="*80)

        # 保存错误报告到文件
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(DATASET_PATH, f'error_report_{timestamp}.txt')

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"错误数据详细报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")

                f.write("错误类型统计:\n")
                for err_type, errors in sorted(error_by_type.items(), key=lambda x: len(x[1]), reverse=True):
                    f.write(f"  • {err_type}: {len(errors)} 条\n")

                f.write("\n详细错误列表:\n")
                f.write(f"{'行号':<8} {'批次范围':<15} {'错误类型':<30} {'需求内容'}\n")
                f.write("-" * 100 + "\n")

                for err in self.error_details:
                    f.write(f"{err['row_number']:<8} {err['batch_range']:<15} "
                           f"{err['error_type']:<30} {err['requirement']}\n")

                f.write("\n" + "="*80 + "\n")
                f.write(f"总计: {len(self.error_details)} 条错误\n")

            print(f"\n✓ 错误报告已保存到: {report_file}")
        except Exception as e:
            print(f"\n⚠ 保存错误报告失败: {e}")

    def process_batch(self, requirements_batch, start_idx, batch_num):
        """
        【核心方法】处理一个批次(10条需求)
        继承generate.py的稳定逻辑 + 错误检测
        """
        print(f"\n{'=' * 60}")
        print(f"🔧 修复批次 #{batch_num}: 第 {start_idx + 1}-{start_idx + len(requirements_batch)} 条")
        print(f"{'=' * 60}")

        req_text = ""
        for i, req in enumerate(requirements_batch, 1):
            req_text += f"{i}. {req}\n\n"

        prompt = SYSTEM_PROMPT.format(
            count=len(requirements_batch),
            requirements=req_text
        )

        max_retries = 3
        response = None

        for retry_count in range(max_retries):
            if retry_count > 0:
                print(f"\n🔄 检测到生成错误,正在重试 ({retry_count}/{max_retries - 1})...")
                time.sleep(3)

            if not self.send_prompt(prompt):
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(requirements_batch)}",
                        'error': '发送失败'
                    })
                    return [None] * len(requirements_batch)

            if not self.wait_for_response_complete():
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(requirements_batch)}",
                        'error': '等待超时'
                    })
                    return [None] * len(requirements_batch)

            response = self.extract_response()
            print(f"\n响应预览: {response[:200]}...\n")

            # 检测错误响应
            error_keywords = [
                "Something went wrong",
                "生成响应时出错",
                "出现错误",
                "error occurred",
                "failed to generate",
                "请尝试等待一会儿",
                "新建一个对话"
            ]

            is_error_response = any(keyword in response for keyword in error_keywords)

            if is_error_response:
                print(f"  ⚠️ 检测到生成错误: {response[:100]}")
                if retry_count < max_retries - 1:
                    print(f"  ↻ 将在3秒后重新发送...")
                    continue
                else:
                    print(f"  ✗ 已达到最大重试次数({max_retries}),放弃本批次")
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(requirements_batch)}",
                        'error': f'生成错误(重试{max_retries}次后失败)'
                    })
                    return [None] * len(requirements_batch)
            else:
                print(f"  ✓ 响应正常,准备解析")
                break

        # 解析指令
        instructions = self.parse_instructions(response, len(requirements_batch))

        if len(instructions) != len(requirements_batch):
            print(f"  ⚠ 警告: 期望{len(requirements_batch)}条,实际{len(instructions)}条")
            while len(instructions) < len(requirements_batch):
                instructions.append(None)
            instructions = instructions[:len(requirements_batch)]

        return instructions

    def repair_file(self, csv_path):
        """修复单个文件"""
        print(f"\n{'#' * 60}")
        print(f"# 处理文件: {os.path.basename(csv_path)}")
        print(f"{'#' * 60}")

        # 读取文件
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

        # 确保Instruction列存在
        if 'Instruction' not in df.columns:
            df['Instruction'] = ''
        df['Instruction'] = df['Instruction'].astype(str)
        df.loc[df['Instruction'] == 'nan', 'Instruction'] = ''

        # ✨【核心】检测需要修复的批次
        error_batches = self.detect_error_batches(df)

        if not error_batches:
            print("\n✓ 未发现需要修复的错误数据")
            # 仍然打印详细报告(如果有的话)
            self.print_error_report()
            return 0

        # ✨ 打印详细错误报告
        self.print_error_report()

        # 询问是否继续
        print(f"\n⚠️ 发现 {len(error_batches)} 个错误批次,共约 {len(error_batches) * BATCH_SIZE} 条数据需要修复")
        user_input = input("是否继续修复? (y/n): ").strip().lower()
        if user_input != 'y':
            print("❌ 用户取消修复")
            return 0

        # 开启新对话
        self.start_new_chat()

        # ✨【核心】逐批次修复
        repaired_batches = 0
        for batch_num, start_idx, requirements_batch in error_batches:
            # 处理批次
            instructions = self.process_batch(requirements_batch, start_idx, batch_num)

            # 写入结果
            for i, instruction in enumerate(instructions):
                row_idx = start_idx + i
                if instruction:
                    df.at[row_idx, 'Instruction'] = instruction
                    self.repaired_count += 1
                else:
                    df.at[row_idx, 'Instruction'] = "ERROR: 生成失败"

            repaired_batches += 1

            # 保存进度
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 已保存进度: {start_idx + len(requirements_batch)}/{len(df)}")

            # 每2个批次刷新对话
            if repaired_batches % 2 == 0 and repaired_batches < len(error_batches):
                self.start_new_chat()

        # 最终保存
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 文件修复完成: {os.path.basename(csv_path)}")
        print(f"  修复批次: {repaired_batches}")
        print(f"  修复数据: {self.repaired_count} 条\n")

        return self.repaired_count

    def run(self):
        """主运行函数"""
        start_time = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"{'批次完整性修复系统 v2.0 (增强错误检测)':^60}")
        print(f"{'=' * 60}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"批次大小: {BATCH_SIZE} 条/批")
        print(f"✨ 新功能: 精准检测'ERROR: 生成失败'(支持多种引号)")
        print(f"✨ 新功能: 详细错误数据报告")
        print(f"{'=' * 60}\n")

        try:
            self.init_driver()

            # 让用户选择要修复的文件
            print("\n可用文件:")
            csv_files = [f for f in os.listdir(DATASET_PATH)
                        if f.endswith('.csv') and f.startswith('enhanced_')]

            for i, filename in enumerate(csv_files, 1):
                print(f"  {i}. {filename}")

            print(f"  {len(csv_files) + 1}. 全部文件")

            choice = input(f"\n请选择要修复的文件 (1-{len(csv_files) + 1}): ").strip()

            files_to_process = []
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(csv_files):
                    files_to_process = [csv_files[choice_num - 1]]
                elif choice_num == len(csv_files) + 1:
                    files_to_process = csv_files

            if not files_to_process:
                print("无效选择")
                return

            total_repaired = 0
            for filename in files_to_process:
                csv_path = os.path.join(DATASET_PATH, filename)
                repaired = self.repair_file(csv_path)
                total_repaired += repaired

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
    repairer = BatchRepairer()
    repairer.run()