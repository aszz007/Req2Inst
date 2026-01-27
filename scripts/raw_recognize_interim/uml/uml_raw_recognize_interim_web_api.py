"""
UML用例图网页识别脚本 - Qwen3-VL-235B-A22B版
通过qianwen.com网页端识别UML用例图,生成JSON格式结果

核心功能:
1. 自动化上传图片到千问网页端
2. 发送完整prompt进行识别
3. 提取JSON格式的识别结果
4. 支持增量处理(跳过已处理图片)
5. 批处理:每次对话5张图片
6. 完善的重试机制

作者: 基于image_dataset_generate.py和uml_recognizer_en.py改编
"""

import os
import time
import json
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from datetime import datetime
import re

# ==================== 配置参数 ====================
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
QIANWEN_URL = "https://www.qianwen.com/"

# 路径配置
IMAGE_FOLDER = r"D:\MyPyProject\crowdsourcing_instruction_generator\data\raw\uml_raw\roboflow_uml"
OUTPUT_JSON = r"D:\MyPyProject\crowdsourcing_instruction_generator\scripts\dataset_recognizer\uml_dataset_recognizer\usecase_recognition_results.json"

# 处理参数
BATCH_SIZE = 1  # 每次对话处理5张图片
REFRESH_INTERVAL = 1  # 每5张图片刷新对话(即每批刷新)
TEST_MODE_LIMIT = 5  # 测试模式只处理5张

# 等待时间配置
WAIT_RESPONSE_TIMEOUT = 90  # 等待响应最多90秒(根据您的测试30秒+余量)
RESPONSE_CHECK_INTERVAL = 2  # 每2秒检查一次响应状态
UPLOAD_WAIT_TIME = 3  # 上传后等待时间

# ==================== 完整Prompt模板 ====================
USECASE_RECOGNITION_PROMPT = """Please carefully analyze this Use Case Diagram and output the recognition results in JSON format.

A Use Case Diagram is a type of UML (Unified Modeling Language) diagram used to describe system functions and user interactions. Please identify the following:

1. **actors**: List of actors (typically stick figures or text labels)
   - Each actor includes: name, position (e.g., "left", "right")

2. **use_cases**: List of use cases (typically ovals)
   - Each use case includes: name, description (brief description)

3. **system_boundary**: System boundary
   - Includes: name (system name), is_present (whether boundary box exists)

4. **relationships**: List of relationships
   - Each relationship includes:
     - type ("association", "include", "extend", "generalization")
     - from (starting element)
     - to (ending element)
     - description (relationship description)

5. **overall_description**: Overall description (summarize the system functionality in one paragraph)

Please output strictly in JSON format. Example:
{
  "actors": [
    {"name": "User", "position": "left"},
    {"name": "Administrator", "position": "right"}
  ],
  "use_cases": [
    {"name": "Login System", "description": "User login functionality"},
    {"name": "View Information", "description": "View personal information"}
  ],
  "system_boundary": {
    "name": "Online Shopping System",
    "is_present": true
  },
  "relationships": [
    {"type": "association", "from": "User", "to": "Login System", "description": "User can login"},
    {"type": "include", "from": "View Information", "to": "Login System", "description": "View information includes login"}
  ],
  "overall_description": "This is a use case diagram describing an online shopping system, showing the basic operation flow of users and administrators"
}

If the image is not a use case diagram or cannot be recognized, please explain in overall_description.
Important: Please ensure complete JSON output with all brackets properly closed. Use English for all content."""


# ==================== 主类 ====================
class UseCaseDiagramWebRecognizer:
    """UML用例图网页识别器 - 千问Qwen3-VL-235B-A22B"""

    def __init__(self, test_mode=True):
        self.test_mode = test_mode
        self.driver = None
        self.processed_images = set()  # 已处理的图片路径
        self.results = []
        self.response_count_before_send = 0

    def init_driver(self):
        """初始化Chrome浏览器 - 使用正确的启动策略"""
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

            print(f"\n正在导航到: {QIANWEN_URL}")
            self.driver.get(QIANWEN_URL)
            time.sleep(8)

            print(f"✓ 页面加载完成: {self.driver.title}")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\n✗ 浏览器初始化失败: {e}")
            raise

    def load_existing_results(self):
        """加载已有的识别结果,支持增量处理"""
        if os.path.exists(OUTPUT_JSON):
            try:
                with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)

                # 提取已处理的图片路径
                for result in self.results:
                    if result.get('success', False):
                        img_path = result.get('image_path', '')
                        if img_path:
                            self.processed_images.add(img_path)

                print(f"\n✓ 加载已有结果: {len(self.results)} 条")
                print(f"✓ 已处理图片: {len(self.processed_images)} 张")
                print(f"将跳过这些图片,只处理新图片\n")

            except Exception as e:
                print(f"⚠ 加载已有结果失败: {e}")
                print("将从头开始处理\n")
                self.results = []
        else:
            print("✓ 未找到已有结果文件,将创建新文件\n")

    def find_new_chat_button(self):
        """定位"新对话"按钮"""
        selectors = [
            "button[class*='newChatButton']",
            "//button[contains(@class, 'newChatButton')]",
            "//button[.//span[text()='新对话']]",
        ]

        for selector in selectors:
            try:
                if selector.startswith('//'):
                    button = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, selector))
                    )
                else:
                    button = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )

                if button.is_displayed() and button.is_enabled():
                    return button
            except:
                continue

        raise NoSuchElementException("无法找到'新对话'按钮")

    def start_new_chat(self):
        """点击新对话按钮"""
        print("\n>>> 开启新对话...")
        try:
            button = self.find_new_chat_button()
            self.driver.execute_script("arguments[0].click();", button)
            print("  ✓ 新对话按钮已点击")
            time.sleep(3)
            self.response_count_before_send = 0
            print("  ✓ 新对话已就绪\n")
        except Exception as e:
            print(f"  ✗ 开启新对话失败: {e}")
            raise

    def hover_and_click_upload(self, image_path):
        """改进的图片上传方法 - 确保上传完成"""
        print(f"  📤 上传图片: {os.path.basename(image_path)}")

        try:
            abs_path = os.path.abspath(image_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"文件不存在: {abs_path}")

            print(f"    ✓ 文件路径: {abs_path}")

            file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            if not file_inputs:
                raise Exception("页面上没有找到 file input 元素")

            print(f"    ✓ 找到 {len(file_inputs)} 个 file input")

            target_input = None
            for inp in file_inputs:
                accept_attr = inp.get_attribute("accept")
                if accept_attr and ("image" in accept_attr or ".png" in accept_attr or ".jpg" in accept_attr):
                    target_input = inp
                    print(f"    ✓ 找到图片上传 input: accept={accept_attr}")
                    break

            if not target_input:
                target_input = file_inputs[0]
                print(f"    ⚠ 使用第一个 file input")

            target_input.send_keys(abs_path)
            print(f"    ✓ 文件路径已发送到 input")
            time.sleep(2)

            print(f"    ⏳ 等待图片预览加载...")
            upload_verified = False

            for attempt in range(10):
                try:
                    preview_selectors = [
                        f"img[src*='{os.path.basename(image_path)[:10]}']",
                        "img[alt*='preview']",
                        "img[class*='preview']",
                        "img[class*='upload']",
                        "div[class*='image-preview']",
                        "div[class*='upload-item']",
                    ]

                    for selector in preview_selectors:
                        try:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            if elements and len(elements) > 0:
                                print(f"    ✓ 检测到图片预览 ({selector})")
                                upload_verified = True
                                break
                        except:
                            continue

                    if upload_verified:
                        break

                    filename = os.path.basename(image_path)[:30]
                    if filename in self.driver.page_source:
                        print(f"    ✓ 页面中检测到文件名")
                        upload_verified = True
                        break

                    time.sleep(1)
                except:
                    time.sleep(1)
                    continue

            if not upload_verified:
                print(f"    ⚠ 无法验证上传是否成功,但继续执行...")

            print(f"    ✓ 图片上传完成")
            time.sleep(UPLOAD_WAIT_TIME)

        except Exception as e:
            print(f"    ✗ 上传失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def send_prompt(self, prompt_text):
        """发送prompt - 使用剪贴板粘贴(模拟真实用户行为),带详细日志"""
        print("  📝 发送prompt...")

        try:
            self.response_count_before_send = self.get_current_response_count()
            print(f"\n    📊 发送前页面已有 {self.response_count_before_send} 条回复")

            print("    ⏳ 等待图片预览加载...")
            time.sleep(3)

            # ========== 可重用函数 ==========
            def find_input_box():
                """定位输入框"""
                selectors = [
                    "textarea[placeholder*='问']",
                    "div[contenteditable='true'][role='textbox']",
                    "div[contenteditable='true']",
                    "textarea",
                ]
                for selector in selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for elem in elements:
                            if elem.is_displayed() and elem.is_enabled():
                                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
                                time.sleep(0.2)
                                return elem, selector, elem.tag_name.lower()
                    except:
                        continue
                return None, None, None

            def get_input_value(input_elem, tag_name):
                """获取输入框当前值"""
                if tag_name == "textarea":
                    return input_elem.get_attribute("value") or ""
                else:
                    return input_elem.text or input_elem.get_attribute("textContent") or ""

            # ========== 步骤1: 定位输入框 ==========
            input_box, used_selector, tag_name = find_input_box()
            if not input_box:
                raise Exception("无法找到输入框")

            print(f"    ✓ 找到输入框: {used_selector}")

            # ========== 步骤2: 使用剪贴板粘贴文本(最接近真实用户行为)==========
            try:
                import pyperclip
                from selenium.webdriver.common.keys import Keys
                from selenium.webdriver.common.action_chains import ActionChains

                try:
                    input_box.clear()
                    time.sleep(0.3)
                except:
                    pass

                input_box.click()
                time.sleep(0.5)

                pyperclip.copy(prompt_text)
                print(f"    ✓ 文本已复制到剪贴板 ({len(prompt_text)} 字符)")

                actions = ActionChains(self.driver)
                actions.key_down(Keys.CONTROL).send_keys('v').key_up(Keys.CONTROL).perform()
                print(f"    ✓ 执行Ctrl+V粘贴")

                time.sleep(2)

                input_box, _, tag_name = find_input_box()
                current_value = get_input_value(input_box, tag_name)

                if current_value and len(current_value) > 100:
                    print(f"    ✓ 粘贴成功 ({len(current_value)} 字符)")
                    print(f"    ✓ 内容可见且完整")
                else:
                    raise Exception(f"粘贴失败或内容不完整 (只有{len(current_value or '')}字符)")

            except ImportError:
                print("    ⚠ pyperclip未安装,使用备用方法...")
                print("    💡 建议安装: pip install pyperclip")

                input_box.clear()
                input_box.click()
                time.sleep(0.5)

                chunk_size = 100
                for i in range(0, len(prompt_text), chunk_size):
                    chunk = prompt_text[i:i + chunk_size]
                    input_box.send_keys(chunk)
                    time.sleep(0.1)

                print(f"    ✓ 分段输入完成")
                time.sleep(1)

                input_box, _, tag_name = find_input_box()
                current_value = get_input_value(input_box, tag_name)
                if not current_value or len(current_value) < 100:
                    raise Exception("备用方案也失败了")

            # ========== 步骤3: 等待UI更新 ==========
            time.sleep(1.5)

            # ========== 步骤4: 再次验证文本存在(防止UI重渲染丢失)==========
            input_box, _, tag_name = find_input_box()
            verify_value = get_input_value(input_box, tag_name)

            if len(verify_value) < 100:
                print(f"    ⚠ 检测到文本丢失,重新粘贴...")

                input_box.click()
                time.sleep(0.3)

                try:
                    import pyperclip
                    from selenium.webdriver.common.keys import Keys
                    from selenium.webdriver.common.action_chains import ActionChains

                    pyperclip.copy(prompt_text)
                    actions = ActionChains(self.driver)
                    actions.key_down(Keys.CONTROL).send_keys('v').key_up(Keys.CONTROL).perform()
                    time.sleep(2)

                    input_box, _, tag_name = find_input_box()
                    verify_value = get_input_value(input_box, tag_name)
                except:
                    pass

            print(f"    ✓ 最终确认文本存在 ({len(verify_value)} 字符)")

            # ========== 步骤5: 发送 - 使用更可靠的Enter键方法 ==========
            print("    📤 准备发送...")

            input_box, _, tag_name = find_input_box()
            input_box.click()
            time.sleep(0.5)

            # 方法1: 直接在输入框上按Enter (最可靠)
            try:
                from selenium.webdriver.common.keys import Keys
                input_box.send_keys(Keys.ENTER)
                print(f"    ✓ 已发送Enter键(方法1: send_keys)")
                time.sleep(3)

                verify_input, _, verify_tag = find_input_box()
                if verify_input:
                    verify_value = get_input_value(verify_input, verify_tag)
                    if not verify_value or len(verify_value.strip()) == 0:
                        print("    ✓ 确认已发送 (输入框已清空)")
                        return True
                    else:
                        print(f"    ⚠ 输入框未清空,尝试方法2...")
                        raise Exception("需要尝试其他方法")

                return True

            except Exception as e1:
                print(f"    ⚠ 方法1失败: {e1}")

                # 方法2: 使用ActionChains按Enter
                try:
                    from selenium.webdriver.common.keys import Keys
                    from selenium.webdriver.common.action_chains import ActionChains

                    input_box, _, _ = find_input_box()
                    input_box.click()
                    time.sleep(0.3)

                    actions = ActionChains(self.driver)
                    actions.send_keys(Keys.ENTER).perform()
                    print(f"    ✓ 已发送Enter键(方法2: ActionChains)")
                    time.sleep(3)

                    verify_input, _, verify_tag = find_input_box()
                    if verify_input:
                        verify_value = get_input_value(verify_input, verify_tag)
                        if not verify_value or len(verify_value.strip()) == 0:
                            print("    ✓ 确认已发送 (输入框已清空)")
                            return True

                    return True

                except Exception as e2:
                    print(f"    ⚠ 方法2失败: {e2}")

                    # 方法3: 查找并点击发送按钮
                    try:
                        print("    🔍 尝试查找发送按钮...")

                        window_height = self.driver.execute_script("return window.innerHeight;")
                        all_buttons = self.driver.find_elements(By.TAG_NAME, "button")

                        candidates = []
                        for btn in all_buttons:
                            try:
                                if not (btn.is_displayed() and btn.is_enabled()):
                                    continue

                                location = btn.location
                                btn_class = btn.get_attribute("class") or ""
                                btn_aria = btn.get_attribute("aria-label") or ""
                                has_svg = len(btn.find_elements(By.TAG_NAME, "svg")) > 0

                                score = 0
                                if location['y'] > window_height * 0.5:
                                    score += 3
                                if has_svg:
                                    score += 2
                                if any(kw in btn_class.lower() for kw in ['send', 'submit', 'arrow', 'up']):
                                    score += 5
                                if any(kw in btn_aria.lower() for kw in ['send', '发送', 'submit']):
                                    score += 5

                                if score >= 3:
                                    candidates.append((btn, score))
                            except:
                                continue

                        if candidates:
                            candidates.sort(key=lambda x: x[1], reverse=True)
                            send_button = candidates[0][0]
                            print(f"    ✓ 找到发送按钮 (得分: {candidates[0][1]})")

                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", send_button)
                            time.sleep(0.5)

                            try:
                                send_button.click()
                                print(f"    ✓ 点击蓝色发送按钮")
                            except:
                                self.driver.execute_script("arguments[0].click();", send_button)
                                print(f"    ✓ JS点击蓝色发送按钮")

                            time.sleep(3)
                            return True
                        else:
                            raise Exception("未找到合适的发送按钮")

                    except Exception as e3:
                        print(f"    ✗ 方法3失败: {e3}")
                        raise Exception("所有发送方法都失败了")

        except Exception as e:
            print(f"    ✗ 发送失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_current_response_count(self):
        """获取当前回复数量 - 增强版,多策略统计,详细日志"""
        try:
            script = """
            // ===== 策略1: 统计所有assistant角色的消息 =====
            let assistantMessages = document.querySelectorAll('[data-message-author-role="assistant"]');
            
            if (assistantMessages.length > 0) {
                // 过滤掉隐藏的和内容太短的
                const valid = Array.from(assistantMessages).filter(el => {
                    // 检查元素是否可见
                    if (el.offsetParent === null) return false;
                    
                    const text = el.textContent || '';
                    // 放宽内容长度要求,因为正在生成的回复可能很短
                    if (text.length < 20) return false;
                    
                    return true;
                });
                
                if (valid.length > 0) {
                    return {
                        count: valid.length, 
                        method: 'assistant-role',
                        details: `找到${assistantMessages.length}个,有效${valid.length}个`
                    };
                }
            }
            
            // ===== 策略2: 统计包含JSON或代码的消息块 =====
            const codeBlocks = document.querySelectorAll('pre, code, div[class*="markdown"]');
            let jsonBlockCount = 0;
            const seenTexts = new Set();
            
            for (const block of codeBlocks) {
                if (block.offsetParent === null) continue;
                
                const text = block.textContent || '';
                // 检查是否包含JSON特征
                if ((text.includes('"actors"') || text.includes('"use_cases"')) && text.length > 100) {
                    // 避免重复计数(同一个回复可能有多个代码块)
                    const signature = text.substring(0, 200);
                    if (!seenTexts.has(signature)) {
                        seenTexts.add(signature);
                        jsonBlockCount++;
                    }
                }
            }
            
            if (jsonBlockCount > 0) {
                return {
                    count: jsonBlockCount,
                    method: 'json-blocks',
                    details: `找到${jsonBlockCount}个JSON块`
                };
            }
            
            // ===== 策略3: 通过文本特征判断(最后的兜底方案) =====
            const bodyText = document.body.textContent || '';
            const actorMatches = (bodyText.match(/"actors"\s*:/g) || []).length;
            const useCaseMatches = (bodyText.match(/"use_cases"\s*:/g) || []).length;
            
            if (actorMatches > 0 && useCaseMatches > 0) {
                // 取两者中较小的值,因为每个完整回复都应该同时包含这两个字段
                const estimatedCount = Math.min(actorMatches, useCaseMatches);
                return {
                    count: estimatedCount,
                    method: 'text-pattern',
                    details: `actors:${actorMatches}, use_cases:${useCaseMatches}`
                };
            }
            
            // ===== 策略4: 宽松的消息容器统计 =====
            const messageContainers = document.querySelectorAll('div[class*="message"], div[class*="response"]');
            let longMessageCount = 0;
            
            for (const container of messageContainers) {
                if (container.offsetParent === null) continue;
                
                const text = container.textContent || '';
                // 降低长度要求,适应正在生成的消息
                if (text.length > 200) {
                    longMessageCount++;
                }
            }
            
            if (longMessageCount > 0) {
                return {
                    count: longMessageCount,
                    method: 'message-containers',
                    details: `找到${longMessageCount}个长消息`
                };
            }
            
            return {count: 0, method: 'none', details: '未检测到回复'};
            """

            result = self.driver.execute_script(script)
            if isinstance(result, dict):
                count = result.get('count', 0)
                method = result.get('method', 'unknown')
                details = result.get('details', '')

                # 详细日志输出,便于调试
                if count > 0:
                    print(f"[回复数:{count},方法:{method},{details}]", end='', flush=True)
                else:
                    print(f"[无回复:{method}]", end='', flush=True)

                return count
            return result if result else 0
        except Exception as e:
            print(f"[计数异常:{str(e)[:30]}]", end='', flush=True)
            return 0

    def check_stop_button_exists(self):
        """精确检测停止按钮 - 使用JavaScript增强检测"""
        try:
            script = """
            const buttons = Array.from(document.querySelectorAll('button'));
            const viewportHeight = window.innerHeight;

            for (const btn of buttons) {
                const rect = btn.getBoundingClientRect();
                if (rect.top < viewportHeight * 0.6) continue;

                const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                if (ariaLabel.includes('stop') || ariaLabel.includes('停止')) {
                    return 'stop_found';
                }

                if (rect.width > 0 && Math.abs(rect.width - rect.height) < 10) {
                    const svg = btn.querySelector('svg');
                    if (svg) {
                        const svgHTML = svg.outerHTML.toLowerCase();

                        if (svgHTML.includes('<rect') || 
                            (svgHTML.includes('<circle') && !svgHTML.includes('polygon'))) {
                            return 'stop_found';
                        }

                        if (svgHTML.includes('polygon') || svgHTML.includes('arrow')) {
                            return 'send_found';
                        }
                    }
                }
            }

            return 'none';
            """

            result = self.driver.execute_script(script)

            if result == 'stop_found':
                print(f"[找到停止按钮→还在生成]", end='', flush=True)
                return True
            elif result == 'send_found':
                print(f"[找到发送按钮→已完成]", end='', flush=True)
                return False
            else:
                print(f"[无明确按钮]", end='', flush=True)
                return self._check_stop_button_backup()

        except Exception as e:
            print(f"[JS检测异常:{str(e)[:20]}]", end='', flush=True)
            return self._check_stop_button_backup()

    def get_latest_response_text(self):
        """获取最新回复的文本内容 - 增强版,确保获取最后一条回复"""
        try:
            script = """
            // ===== 策略1: 获取最后一个assistant消息 =====
            const assistantMessages = document.querySelectorAll('[data-message-author-role="assistant"]');
            if (assistantMessages.length > 0) {
                // 从后向前查找第一个可见的
                for (let i = assistantMessages.length - 1; i >= 0; i--) {
                    const elem = assistantMessages[i];
                    if (elem.offsetParent !== null) {
                        const text = elem.textContent || elem.innerText || '';
                        // 放宽长度要求,因为正在生成的回复可能很短
                        if (text.length > 20) {
                            return {
                                text: text, 
                                method: 'assistant-last',
                                index: i,
                                total: assistantMessages.length,
                                length: text.length
                            };
                        }
                    }
                }
            }
            
            // ===== 策略2: 查找最后一个包含JSON的代码块 =====
            const codeBlocks = Array.from(document.querySelectorAll('pre, code, div[class*="markdown"]'));
            for (let i = codeBlocks.length - 1; i >= 0; i--) {
                const block = codeBlocks[i];
                if (block.offsetParent === null) continue;
                
                const text = block.textContent || block.innerText || '';
                // 检查是否包含JSON特征
                if ((text.includes('"actors"') || text.includes('"use_cases"')) && text.length > 100) {
                    return {
                        text: text,
                        method: 'code-block-last',
                        length: text.length
                    };
                }
            }
            
            // ===== 策略3: 查找最后一个长消息 =====
            const selectors = [
                'div[class*="message"]',
                'div[class*="response"]',
                '.markdown-body'
            ];

            for (const selector of selectors) {
                const elements = Array.from(document.querySelectorAll(selector));
                // 从后向前查找
                for (let i = elements.length - 1; i >= 0; i--) {
                    const elem = elements[i];
                    if (elem.offsetParent === null) continue;
                    
                    const text = elem.textContent || elem.innerText || '';
                    // 降低长度要求
                    if (text.length > 100) {
                        return {
                            text: text,
                            method: selector + '-last',
                            length: text.length
                        };
                    }
                }
            }

            return {text: '', method: 'none', length: 0};
            """

            result = self.driver.execute_script(script)
            if isinstance(result, dict):
                text = result.get('text', '')
                method = result.get('method', 'unknown')
                length = result.get('length', 0)

                # 详细日志
                if text:
                    extra_info = ""
                    if 'index' in result and 'total' in result:
                        extra_info = f",第{result['index']+1}/{result['total']}条"
                    print(f"[文本:{length}字符,{method}{extra_info}]", end='', flush=True)
                else:
                    print(f"[无文本:{method}]", end='', flush=True)

                return text
            return result if result else ""
        except Exception as e:
            print(f"[获取异常:{str(e)[:30]}]", end='', flush=True)
            return ""

    def check_if_still_generating(self):
        """检测是否还在生成 - 基于文本增长的可靠方法,带日志"""
        try:
            first_text = self.get_latest_response_text()
            first_len = len(first_text)

            if first_len < 10:
                print(f"[无内容:{first_len}]", end='', flush=True)
                return True, "无内容"

            time.sleep(1.2)

            second_text = self.get_latest_response_text()
            second_len = len(second_text)

            growth = second_len - first_len

            if growth > 0:
                # 只要有任何增长（哪怕1个字符）都认为还在生成
                print(f"[增长+{growth}→生成中]", end='', flush=True)
                return True, f"文本增长{growth}"
            else:
                print(f"[稳定:{first_len}字符→完成]", end='', flush=True)
                return False, "文本无增长"

        except Exception as e:
            print(f"[检测异常:{str(e)[:30]}→完成]", end='', flush=True)
            return False, f"异常:{str(e)[:20]}"

    def _check_stop_button_backup(self):
        """备用方法:通过Python Selenium检测停止按钮"""
        try:
            window_height = self.driver.execute_script("return window.innerHeight;")
            all_buttons = self.driver.find_elements(By.TAG_NAME, "button")

            for btn in all_buttons:
                try:
                    if not btn.is_displayed():
                        continue

                    location = btn.location
                    size = btn.size

                    if location['y'] < window_height * 0.6:
                        continue

                    width = size.get('width', 0)
                    height = size.get('height', 0)
                    if width > 0 and abs(width - height) < 10:
                        svgs = btn.find_elements(By.TAG_NAME, "svg")
                        if svgs:
                            is_enabled = btn.is_enabled()
                            if is_enabled:
                                print(f"[备用:找到圆形按钮→还在生成]", end='', flush=True)
                                return True
                except:
                    continue

            print(f"[备用:未找到停止按钮→已完成]", end='', flush=True)
            return False

        except Exception as e:
            print(f"[备用检测异常]", end='', flush=True)
            return False

    def check_stop_button_simple(self):
        """简化的停止按钮检测 - 只作为辅助判断"""
        try:
            window_height = self.driver.execute_script("return window.innerHeight;")
            buttons = self.driver.find_elements(By.TAG_NAME, "button")

            for btn in buttons:
                try:
                    if not btn.is_displayed():
                        continue

                    location = btn.location
                    size = btn.size

                    if location['y'] < window_height * 0.6:
                        continue

                    width = size.get('width', 0)
                    height = size.get('height', 0)

                    if width > 20 and abs(width - height) < 10:
                        aria_label = btn.get_attribute("aria-label") or ""

                        if "停止" in aria_label or "stop" in aria_label.lower():
                            return True
                        elif "发送" in aria_label or "send" in aria_label.lower():
                            return False

                        if btn.is_enabled() and location['x'] > window_height * 0.8:
                            return True

                except:
                    continue

            return None
        except:
            return None

    def check_response_still_updating(self):
        """检测是否还在生成 - 仅基于文本增长,带日志"""
        try:
            print(f"[检测]", end='', flush=True)

            is_generating, reason = self.check_if_still_generating()

            return is_generating

        except Exception as e:
            print(f"[异常:{str(e)[:20]}→完成]", end='', flush=True)
            return False

    def wait_for_response_complete(self):
        """等待响应完成 - 优化版,带详细调试日志"""
        print("  ⏳ 等待生成...", end='', flush=True)
        start_time = time.time()

        print(f" [等待回复][发送前有{self.response_count_before_send}条]", end='', flush=True)

        max_wait = int(WAIT_RESPONSE_TIMEOUT / RESPONSE_CHECK_INTERVAL)
        for i in range(max_wait):
            count = self.get_current_response_count()

            # 详细日志:显示当前检测到的数量 vs 期望的数量
            if count > self.response_count_before_send:
                # 二次确认,避免误判
                time.sleep(2)
                recheck = self.get_current_response_count()
                if recheck > self.response_count_before_send:
                    elapsed = int(time.time() - start_time)
                    print(f" ✓ [新回复出现:当前{recheck}条>发送前{self.response_count_before_send}条,用时{elapsed}s]", end='', flush=True)
                    break
                else:
                    # 数量又回退了,可能是误判
                    print(f"[数量回退:{count}→{recheck},继续等待]", end='', flush=True)
            elif count == self.response_count_before_send:
                # 数量未变化,继续等待
                if (i + 1) % 3 == 0:
                    elapsed = int(time.time() - start_time)
                    print(f"[{elapsed}s,仍为{count}条]", end='', flush=True)
            else:
                # 数量反而减少了(这种情况很少见,可能是页面刷新)
                print(f"[异常:当前{count}<发送前{self.response_count_before_send}]", end='', flush=True)

            time.sleep(RESPONSE_CHECK_INTERVAL)
        else:
            print(f" ✗ 超时(最终检测到{count}条,期望>{self.response_count_before_send}条)")
            return False

        print(" [检测完成]\n", end='', flush=True)

        stable_count = 0
        required = 2

        for check_idx in range(40):
            print(f"  [{check_idx + 1}]", end='', flush=True)

            is_updating = self.check_response_still_updating()

            if is_updating:
                stable_count = 0
            else:
                stable_count += 1
                print(f"[完成{stable_count}/{required}]", end='', flush=True)

                if stable_count >= required:
                    elapsed = int(time.time() - start_time)
                    print(f"\n  ✓ 完成[{elapsed}s]")
                    return True

            print()
            time.sleep(1.5)

        elapsed = int(time.time() - start_time)
        print(f"\n  ✓ 完成(上限)[{elapsed}s]")
        return True

    def extract_json_response(self, image_path):
        """提取JSON响应 - 增强版,确保获取最新回复,智能过滤用户prompt"""
        try:
            print(f"\n  📊 提取响应内容...")

            script = """
            // ===== 优先策略: 查找最后一个包含完整JSON的独立代码块 =====
            // 这是最可靠的方法,因为每个回复的JSON都在单独的代码块里
            const codeBlocks = Array.from(document.querySelectorAll('pre, code'));
            for (let i = codeBlocks.length - 1; i >= 0; i--) {
                const block = codeBlocks[i];
                if (block.offsetParent === null) continue;
                
                const text = block.textContent || block.innerText || '';
                // 必须包含JSON关键字段且足够长
                if (text.includes('"actors"') && text.includes('"use_cases"') && text.length > 200) {
                    return {
                        text: text,
                        method: 'code-block-last',
                        length: text.length
                    };
                }
            }
            
            // ===== 备用策略1: 获取最后一个assistant消息(如果有明确的assistant标记) =====
            const assistantMessages = document.querySelectorAll('[data-message-author-role="assistant"]');
            if (assistantMessages.length > 0) {
                // 从后向前查找
                for (let i = assistantMessages.length - 1; i >= 0; i--) {
                    const elem = assistantMessages[i];
                    if (elem.offsetParent !== null) {
                        const text = elem.textContent || elem.innerText || '';
                        // 确保包含JSON内容
                        if (text.includes('"actors"') && text.length > 200) {
                            return {
                                text: text, 
                                method: 'assistant-last',
                                index: i + 1,
                                total: assistantMessages.length,
                                length: text.length
                            };
                        }
                    }
                }
            }
            
            // ===== 备用策略2: 查找最后一个长消息 =====
            const selectors = [
                'div[class*="message"]',
                'div[class*="response"]',
                '.markdown-body'
            ];

            for (const selector of selectors) {
                const elements = Array.from(document.querySelectorAll(selector));
                for (let i = elements.length - 1; i >= 0; i--) {
                    const elem = elements[i];
                    if (elem.offsetParent === null) continue;
                    
                    const text = elem.textContent || elem.innerText || '';
                    if (text.length > 100) {
                        return {
                            text: text,
                            method: selector + '-last',
                            length: text.length
                        };
                    }
                }
            }

            return {text: '', method: 'none', length: 0};
            """

            result = self.driver.execute_script(script)

            response_text = ""
            method = "unknown"

            if isinstance(result, dict):
                response_text = result.get('text', '')
                method = result.get('method', 'unknown')
                length = result.get('length', 0)

                # 详细日志
                extra_info = ""
                if 'index' in result and 'total' in result:
                    extra_info = f" (第{result['index']}/{result['total']}条)"
                print(f"  ✓ 获取到响应 ({length} 字符, 方法:{method}{extra_info})")
            else:
                response_text = result or ""
                print(f"  ✓ 获取到响应 ({len(response_text)} 字符)")

            if not response_text or len(response_text) < 50:
                print(f"  ✗ 无法获取响应文本")
                return None

            # ========== 第一步：先去除用户prompt（包括其中的示例JSON） ==========
            # 智能过滤:如果响应以用户prompt开头,提取后面的内容
            if response_text.startswith('Please carefully analyze'):
                print(f"  🔧 检测到用户prompt,先进行分离...")

                # 策略1: 查找prompt的结束标志(示例后的说明文字)
                prompt_end_markers = [
                    "If the image is not a use case diagram",
                    "Important: Please ensure complete JSON",
                    "Use English for all content"
                ]

                prompt_end_pos = -1
                for marker in prompt_end_markers:
                    pos = response_text.find(marker)
                    if pos > 0:
                        # 找到标志后,继续找这个标志所在段落的结束
                        # 通常是两个连续换行,或者下一个{
                        remaining = response_text[pos + len(marker):]
                        # 跳过标志文本后的内容,找到真正的JSON开始
                        next_json = remaining.find('{')
                        if next_json > 0:
                            prompt_end_pos = pos + len(marker) + next_json
                            print(f"  ✓ 找到prompt结束标志: '{marker[:30]}...'")
                            break

                if prompt_end_pos > 0:
                    response_text = response_text[prompt_end_pos:]
                    print(f"  ✓ 成功去除prompt,剩余 {len(response_text)} 字符")
                else:
                    # 策略2: 查找所有JSON块,去掉第一个(很可能是示例)
                    print(f"  ⚠ 未找到prompt结束标志,尝试去除示例JSON...")

                    json_objects = []
                    i = 0
                    while i < len(response_text):
                        if response_text[i] == '{':
                            depth = 0
                            start = i
                            for j in range(i, len(response_text)):
                                if response_text[j] == '{':
                                    depth += 1
                                elif response_text[j] == '}':
                                    depth -= 1
                                    if depth == 0:
                                        block = response_text[start:j + 1]
                                        if ('"actors"' in block and '"use_cases"' in block
                                            and len(block) > 200):
                                            json_objects.append({
                                                'text': block,
                                                'start': start,
                                                'end': j + 1
                                            })
                                        i = j
                                        break
                        i += 1

                    if len(json_objects) >= 2:
                        # 有多个JSON块,第一个很可能是示例,去掉第一个后重新拼接
                        # 从第二个JSON开始的位置截取
                        response_text = response_text[json_objects[1]['start']:]
                        print(f"  ✓ 去除第1个JSON(示例),剩余 {len(response_text)} 字符")
                    else:
                        print(f"  ⚠ 只找到 {len(json_objects)} 个JSON,保持原文本")

            # ========== 第二步：在去除prompt后的文本中查找JSON ==========
            # 批处理场景：因为已经去除了prompt，所以直接查找最后一个JSON即可
            print(f"  🔍 在处理后的文本中查找JSON...")

            json_objects = []
            i = 0
            while i < len(response_text):
                if response_text[i] == '{':
                    depth = 0
                    start = i
                    for j in range(i, len(response_text)):
                        if response_text[j] == '{':
                            depth += 1
                        elif response_text[j] == '}':
                            depth -= 1
                            if depth == 0:
                                block = response_text[start:j + 1]
                                # 只保留包含 actors 和 use_cases 的块
                                if ('"actors"' in block and '"use_cases"' in block
                                    and len(block) > 200):
                                    json_objects.append(block)
                                i = j
                                break
                i += 1

            if json_objects:
                # 取最后一个JSON（最新生成的）
                response_text = json_objects[-1]
                print(f"  ✓ 找到 {len(json_objects)} 个JSON,使用最后一个 (长度: {len(response_text)} 字符)")
            else:
                print(f"  ⚠ 未找到有效JSON,使用原文本")
            if response_text.startswith('Please carefully analyze'):
                print(f"  ⚠ 检测到用户prompt在响应开头,尝试分离...")

                # 策略1: 查找prompt的结束标志(示例后的说明文字)
                prompt_end_markers = [
                    "If the image is not a use case diagram",
                    "Important: Please ensure complete JSON",
                    "Use English for all content"
                ]

                prompt_end_pos = -1
                for marker in prompt_end_markers:
                    pos = response_text.find(marker)
                    if pos > 0:
                        # 找到标志后,继续找这个标志所在段落的结束
                        # 通常是两个连续换行,或者下一个{
                        remaining = response_text[pos + len(marker):]
                        # 跳过标志文本后的内容,找到真正的JSON开始
                        next_json = remaining.find('{')
                        if next_json > 0:
                            prompt_end_pos = pos + len(marker) + next_json
                            print(f"  ✓ 找到prompt结束标志: '{marker[:30]}...'")
                            break

                if prompt_end_pos > 0:
                    response_text = response_text[prompt_end_pos:]
                    print(f"  ✓ 成功分离,剩余 {len(response_text)} 字符")
                else:
                    # 策略2: 查找所有JSON块,取最后一个(最有可能是真实回复)
                    print(f"  ⚠ 未找到prompt结束标志,尝试查找多个JSON块...")

                    # 找到所有完整的JSON对象(匹配的{})
                    json_blocks = []
                    i = 0
                    while i < len(response_text):
                        if response_text[i] == '{':
                            # 尝试提取这个JSON块
                            depth = 0
                            start = i
                            for j in range(i, len(response_text)):
                                if response_text[j] == '{':
                                    depth += 1
                                elif response_text[j] == '}':
                                    depth -= 1
                                    if depth == 0:
                                        # 找到完整的JSON块
                                        block = response_text[start:j + 1]
                                        if len(block) > 100:  # 过滤太短的块
                                            json_blocks.append(block)
                                        i = j
                                        break
                        i += 1

                    if len(json_blocks) >= 2:
                        # 有多个JSON块,第一个很可能是示例,取最后一个
                        print(f"  ✓ 找到 {len(json_blocks)} 个JSON块,使用最后一个")
                        response_text = json_blocks[-1]
                        print(f"  ✓ 提取最后JSON块,长度 {len(response_text)} 字符")
                    else:
                        print(f"  ⚠ 只找到 {len(json_blocks)} 个JSON块,继续使用原文本")

            # 显示前200字符预览
            preview = response_text[:200].replace('\n', ' ')
            print(f"  🔍 内容预览: {preview}...")

            json_result = None

            # 方法1: 直接解析
            try:
                json_result = json.loads(response_text.strip())
                print(f"  ✓ 直接解析成功")
            except:
                pass

            # 方法2: 提取```json代码块
            if not json_result:
                try:
                    patterns = [
                        r'```json\s*([\s\S]*?)\s*```',
                        r'```\s*([\s\S]*?)\s*```'
                    ]

                    for pattern in patterns:
                        matches = re.findall(pattern, response_text)
                        if matches:
                            for match in matches:
                                try:
                                    json_result = json.loads(match.strip())
                                    print(f"  ✓ 从代码块提取成功")
                                    break
                                except:
                                    continue
                        if json_result:
                            break
                except:
                    pass

            # 方法3: 查找第一个完整的JSON对象
            if not json_result:
                try:
                    json_pattern = r'\{[\s\S]*\}'
                    match = re.search(json_pattern, response_text)
                    if match:
                        json_str = match.group(0)
                        try:
                            json_result = json.loads(json_str)
                            print(f"  ✓ 提取JSON对象成功")
                        except:
                            open_braces = json_str.count('{')
                            close_braces = json_str.count('}')
                            if open_braces > close_braces:
                                json_str += '}' * (open_braces - close_braces)
                            try:
                                json_result = json.loads(json_str)
                                print(f"  ✓ 修复后解析成功")
                            except:
                                pass
                except:
                    pass

            # 方法4: 逐行查找JSON关键字段并重构
            if not json_result:
                try:
                    print(f"  ⚠ 尝试从文本重构JSON...")

                    if '"actors"' in response_text and '"use_cases"' in response_text:
                        actors_match = re.search(r'"actors"\s*:\s*(\[[\s\S]*?\])', response_text)
                        use_cases_match = re.search(r'"use_cases"\s*:\s*(\[[\s\S]*?\])', response_text)
                        system_match = re.search(r'"system_boundary"\s*:\s*(\{[\s\S]*?\})', response_text)
                        rel_match = re.search(r'"relationships"\s*:\s*(\[[\s\S]*?\])', response_text)
                        desc_match = re.search(r'"overall_description"\s*:\s*"([^"]*)"', response_text)

                        reconstructed = {
                            "actors": json.loads(actors_match.group(1)) if actors_match else [],
                            "use_cases": json.loads(use_cases_match.group(1)) if use_cases_match else [],
                            "system_boundary": json.loads(system_match.group(1)) if system_match else {},
                            "relationships": json.loads(rel_match.group(1)) if rel_match else [],
                            "overall_description": desc_match.group(1) if desc_match else ""
                        }

                        json_result = reconstructed
                        print(f"  ✓ 重构JSON成功")
                except Exception as e:
                    print(f"  ✗ 重构失败: {e}")

            if not json_result:
                print(f"  ✗ 所有JSON提取方法都失败")
                return {
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'success': False,
                    'error': 'JSON解析失败',
                    'raw_response': response_text[:1000],
                    'timestamp': datetime.now().isoformat()
                }

            # 验证必需字段
            required_fields = ['actors', 'use_cases', 'system_boundary', 'relationships', 'overall_description']
            missing = [f for f in required_fields if f not in json_result]

            if missing:
                print(f"  ⚠ 缺少字段: {missing}, 补充默认值")
                for field in missing:
                    if field in ['actors', 'use_cases', 'relationships']:
                        json_result[field] = []
                    elif field == 'system_boundary':
                        json_result[field] = {}
                    else:
                        json_result[field] = ""

            json_result['image_path'] = image_path
            json_result['image_name'] = os.path.basename(image_path)
            json_result['success'] = True
            json_result['timestamp'] = datetime.now().isoformat()

            print(f"  ✓ JSON提取成功")
            print(f"    - actors: {len(json_result.get('actors', []))}")
            print(f"    - use_cases: {len(json_result.get('use_cases', []))}")
            print(f"    - relationships: {len(json_result.get('relationships', []))}")

            return json_result

        except Exception as e:
            print(f"  ✗ 提取异常: {e}")
            import traceback
            traceback.print_exc()

            return {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def process_single_image(self, image_path, is_first_in_chat, max_retries=3):
        """处理单张图片(含重试机制)"""
        print(f"\n{'=' * 70}")
        print(f"处理: {os.path.basename(image_path)}")
        print(f"{'=' * 70}")

        for attempt in range(max_retries):
            if attempt > 0:
                print(f"\n🔄 第 {attempt + 1} 次重试...")
                time.sleep(3)

            try:
                self.hover_and_click_upload(image_path)

                if not self.send_prompt(USECASE_RECOGNITION_PROMPT):
                    if attempt < max_retries - 1:
                        continue
                    return None

                if not self.wait_for_response_complete():
                    if attempt < max_retries - 1:
                        continue
                    return None

                result = self.extract_json_response(image_path)

                if result and result.get('success', False):
                    return result
                elif attempt < max_retries - 1:
                    print("  ⚠ 结果验证失败,准备重试...")
                    continue
                else:
                    return result

            except Exception as e:
                print(f"  ✗ 处理异常: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return {
                        'image_path': image_path,
                        'image_name': os.path.basename(image_path),
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }

        return None

    def process_batch(self, image_paths):
        """处理一批图片(5张)"""
        print(f"\n{'#' * 80}")
        print(f"# 开始新批次: {len(image_paths)} 张图片")
        print(f"{'#' * 80}")

        self.start_new_chat()

        batch_results = []

        for i, img_path in enumerate(image_paths):
            is_first = (i == 0)
            result = self.process_single_image(img_path, is_first)

            if result:
                batch_results.append(result)
                self.results.append(result)
            else:
                error_result = {
                    'image_path': img_path,
                    'image_name': os.path.basename(img_path),
                    'success': False,
                    'error': '处理失败',
                    'timestamp': datetime.now().isoformat()
                }
                batch_results.append(error_result)
                self.results.append(error_result)

        return batch_results

    def save_progress(self):
        """保存进度到JSON文件"""
        try:
            output_dir = os.path.dirname(OUTPUT_JSON)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)

            print(f"\n  💾 进度已保存: {OUTPUT_JSON}")

        except Exception as e:
            print(f"\n  ✗ 保存失败: {e}")

    def run(self):
        """主流程"""
        start_time = datetime.now()

        print(f"\n{'=' * 80}")
        print(f"{'UML用例图网页识别系统 - Qwen3-VL-235B-A22B':^80}")
        print(f"{'=' * 80}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模式: {'测试模式 (5张)' if self.test_mode else '完整模式'}")
        print(f"批次大小: {BATCH_SIZE} 张/批")
        print(f"图片文件夹: {IMAGE_FOLDER}")
        print(f"输出文件: {OUTPUT_JSON}")
        print(f"{'=' * 80}\n")

        try:
            self.init_driver()
            self.load_existing_results()

            image_folder = Path(IMAGE_FOLDER)
            if not image_folder.exists():
                raise FileNotFoundError(f"图片文件夹不存在: {IMAGE_FOLDER}")

            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
            all_images = []
            for ext in image_extensions:
                all_images.extend(image_folder.glob(f"*{ext}"))
                all_images.extend(image_folder.glob(f"*{ext.upper()}"))

            all_images = sorted(list(set(all_images)))

            print(f"✓ 扫描到图片: {len(all_images)} 张")

            pending_images = []
            for img in all_images:
                if str(img) not in self.processed_images:
                    pending_images.append(str(img))

            print(f"✓ 待处理图片: {len(pending_images)} 张\n")

            if len(pending_images) == 0:
                print("✓ 所有图片已处理完成!")
                return

            if self.test_mode:
                pending_images = pending_images[:TEST_MODE_LIMIT]
                print(f"*** 测试模式: 只处理前 {len(pending_images)} 张 ***\n")

            total_images = len(pending_images)
            success_count = 0
            fail_count = 0

            for i in range(0, total_images, BATCH_SIZE):
                batch_images = pending_images[i:i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1

                print(f"\n{'>' * 80}")
                print(f"> 批次 {batch_num}: 图片 {i + 1}-{i + len(batch_images)} / {total_images}")
                print(f"{'>' * 80}")

                batch_results = self.process_batch(batch_images)

                for result in batch_results:
                    if result.get('success', False):
                        success_count += 1
                    else:
                        fail_count += 1

                self.save_progress()

                print(f"\n📊 当前统计: 成功 {success_count} | 失败 {fail_count}")

            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n{'=' * 80}")
            print(f"{'处理完成':^80}")
            print(f"{'=' * 80}")
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"耗时: {duration}")
            print(f"总计处理: {total_images} 张")
            print(f"成功: {success_count} 张")
            print(f"失败: {fail_count} 张")
            print(f"成功率: {success_count / total_images * 100:.1f}%")
            print(f"结果文件: {OUTPUT_JSON}")
            print(f"{'=' * 80}\n")

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
    print("\n" + "=" * 80)
    print("请选择运行模式:")
    print(f"  1. 测试模式 (仅处理前{TEST_MODE_LIMIT}张)")
    print("  2. 完整模式 (处理所有未处理的图片)")
    print("=" * 80)

    while True:
        choice = input("\n请输入选项 (1 或 2): ").strip()
        if choice == "1":
            test_mode = True
            print("\n✓ 已选择: 测试模式")
            break
        elif choice == "2":
            test_mode = False
            print("\n✓ 已选择: 完整模式")
            confirm = input("⚠ 完整模式将处理大量数据,确认继续? (y/n): ").strip().lower()
            if confirm == 'y':
                break
            else:
                print("已取消")
                exit(0)
        else:
            print("✗ 无效选项,请输入 1 或 2")

    recognizer = UseCaseDiagramWebRecognizer(test_mode=test_mode)
    recognizer.run()