"""
UML众包指令批次修复脚本
基于uml_interim_generate_dataset.py的生成逻辑和image_dataset_regenerate.py的修复逻辑
核心特性:
1. 继承稳定的浏览器自动化功能
2. 批次完整性检查:如果批次中有任何ERROR,整个批次重新生成
3. 自动检测需要修复的批次范围
4. 精准检测"ERROR: 生成失败",支持多种引号格式
5. 新增:三段式完整性检查(Definition/Emphasis/Things to Avoid)
6. 新增:句号检查(检测三段式最后是否缺少句号)
7. 详细错误报告,列出每条错误数据及具体问题
8. 使用UML专用Prompt模板和Few-shot示例
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
import json



# ==================== 配置参数 ====================
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DATASET_PATH = r"dataset/uml"
GPT_URL = "https://sass-node1.chatshare.biz/"

# 目标文件
CSV_FILE = "uml_dataset_qwen3_v3.csv"

BATCH_SIZE = 1  # 批次大小(与首次生成保持一致)
WAIT_NEW_RESPONSE_TIMEOUT = 60  # 等待新回复最多60秒
CONTENT_STABLE_CHECKS = 3  # 内容稳定性检查次数

# ==================== 10个领域的优质示例库 (从uml_interim_generate_dataset.py复制) ====================
DOMAIN_EXAMPLES = {
    "ecommerce": {
        "json": """{
  "actors": [{"name": "Customer"}, {"name": "Payment Gateway"}, {"name": "Inventory System"}],
  "use_cases": [
    {"name": "Place Order", "description": "Customer places an order"},
    {"name": "Verify Stock", "description": "Check product availability"},
    {"name": "Process Payment", "description": "Handle payment transaction"},
    {"name": "Send Confirmation", "description": "Email order confirmation"}
  ],
  "relationships": [
    {"type": "association", "from": "Customer", "to": "Place Order"},
    {"type": "include", "from": "Place Order", "to": "Verify Stock"},
    {"type": "include", "from": "Place Order", "to": "Process Payment"},
    {"type": "extend", "from": "Place Order", "to": "Send Confirmation"}
  ],
  "overall_description": "E-commerce order placement system with mandatory stock verification and payment processing, plus optional email confirmation."
}""",
        "instruction": """Definition: In this task, implement the "Place Order" workflow where Customer interacts with the system, ensuring mandatory stock verification and payment processing steps are completed.
Emphasis & Caution: You MUST execute "Verify Stock" and "Process Payment" as required prerequisites (include relationships) before finalizing the order. "Send Confirmation" is a conditional extension that triggers upon successful order completion.
Things to Avoid: Do not use actor position metadata to determine business logic or workflow sequence. Do not implement UI layout based on position values."""
    },

    "authentication": {
        "json": """{
  "actors": [{"name": "User"}, {"name": "OAuth Provider"}, {"name": "Email Service"}],
  "use_cases": [
    {"name": "Login", "description": "User authentication"},
    {"name": "Validate Credentials", "description": "Verify username and password"},
    {"name": "Generate Token", "description": "Create session token"},
    {"name": "Send Verification Email", "description": "Email verification for new devices"}
  ],
  "relationships": [
    {"type": "association", "from": "User", "to": "Login"},
    {"type": "include", "from": "Login", "to": "Validate Credentials"},
    {"type": "include", "from": "Login", "to": "Generate Token"},
    {"type": "extend", "from": "Login", "to": "Send Verification Email"}
  ],
  "overall_description": "User authentication system with mandatory credential validation and token generation, plus optional email verification for new devices."
}""",
        "instruction": """Definition: In this task, implement the "Login" authentication workflow where User interacts with the system, ensuring mandatory credential validation and token generation.
Emphasis & Caution: You MUST enforce "Validate Credentials" and "Generate Token" as required steps (include relationships) that execute automatically during login. "Send Verification Email" is a conditional extension triggered when login occurs from a new device.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "content_management": {
        "json": """{
  "actors": [{"name": "Author"}, {"name": "Editor"}, {"name": "Publisher"}],
  "use_cases": [
    {"name": "Create Article", "description": "Author creates content"},
    {"name": "Submit for Review", "description": "Submit to editorial queue"},
    {"name": "Approve Content", "description": "Editor approves article"},
    {"name": "Publish", "description": "Make content live"}
  ],
  "relationships": [
    {"type": "association", "from": "Author", "to": "Create Article"},
    {"type": "include", "from": "Create Article", "to": "Submit for Review"},
    {"type": "association", "from": "Editor", "to": "Approve Content"},
    {"type": "extend", "from": "Approve Content", "to": "Publish"}
  ],
  "overall_description": "Content management system where authors create articles with mandatory review submission, editors approve, and optional immediate publishing."
}""",
        "instruction": """Definition: In this task, implement the "Create Article" workflow where Author creates content with mandatory review submission, and the "Approve Content" process where Editor reviews articles with optional publishing.
Emphasis & Caution: You MUST enforce "Submit for Review" as a required step (include relationship) that executes automatically after article creation. "Publish" is a conditional extension of approval that triggers when immediate publishing is selected.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "social_interaction": {
        "json": """{
  "actors": [{"name": "User"}, {"name": "Follower"}, {"name": "Notification Service"}],
  "use_cases": [
    {"name": "Create Post", "description": "User creates a post"},
    {"name": "Validate Content", "description": "Check for prohibited content"},
    {"name": "Notify Followers", "description": "Send notifications to followers"},
    {"name": "Generate Thumbnail", "description": "Create image preview"}
  ],
  "relationships": [
    {"type": "association", "from": "User", "to": "Create Post"},
    {"type": "include", "from": "Create Post", "to": "Validate Content"},
    {"type": "extend", "from": "Create Post", "to": "Notify Followers"},
    {"type": "extend", "from": "Create Post", "to": "Generate Thumbnail"}
  ],
  "overall_description": "Social media post creation system with mandatory content validation and optional follower notifications and thumbnail generation."
}""",
        "instruction": """Definition: In this task, implement the "Create Post" workflow where User creates social media content with mandatory content validation.
Emphasis & Caution: You MUST enforce "Validate Content" as a required step (include relationship) that executes before post publication. "Notify Followers" and "Generate Thumbnail" are conditional extensions that trigger based on user preferences or content type.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "customer_service": {
        "json": """{
  "actors": [{"name": "Customer"}, {"name": "Support Agent"}, {"name": "Ticketing System"}],
  "use_cases": [
    {"name": "Create Ticket", "description": "Customer creates support ticket"},
    {"name": "Assign Category", "description": "Categorize the issue"},
    {"name": "Set Priority", "description": "Determine urgency level"},
    {"name": "Escalate Issue", "description": "Route to senior support"}
  ],
  "relationships": [
    {"type": "association", "from": "Customer", "to": "Create Ticket"},
    {"type": "include", "from": "Create Ticket", "to": "Assign Category"},
    {"type": "include", "from": "Create Ticket", "to": "Set Priority"},
    {"type": "extend", "from": "Create Ticket", "to": "Escalate Issue"}
  ],
  "overall_description": "Customer support ticket system with mandatory categorization and priority setting, plus optional escalation for complex issues."
}""",
        "instruction": """Definition: In this task, implement the "Create Ticket" workflow where Customer submits support requests with mandatory category assignment and priority setting.
Emphasis & Caution: You MUST enforce "Assign Category" and "Set Priority" as required steps (include relationships) that execute during ticket creation. "Escalate Issue" is a conditional extension that triggers when the issue meets escalation criteria.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "data_analysis": {
        "json": """{
  "actors": [{"name": "Analyst"}, {"name": "Data Warehouse"}, {"name": "Reporting Engine"}],
  "use_cases": [
    {"name": "Run Analysis", "description": "Execute data analysis"},
    {"name": "Fetch Data", "description": "Retrieve data from warehouse"},
    {"name": "Generate Report", "description": "Create analysis report"},
    {"name": "Export to CSV", "description": "Export results to CSV"}
  ],
  "relationships": [
    {"type": "association", "from": "Analyst", "to": "Run Analysis"},
    {"type": "include", "from": "Run Analysis", "to": "Fetch Data"},
    {"type": "include", "from": "Run Analysis", "to": "Generate Report"},
    {"type": "extend", "from": "Run Analysis", "to": "Export to CSV"}
  ],
  "overall_description": "Data analysis system with mandatory data fetching and report generation, plus optional CSV export."
}""",
        "instruction": """Definition: In this task, implement the "Run Analysis" workflow where Analyst executes data analysis with mandatory data retrieval and report generation.
Emphasis & Caution: You MUST enforce "Fetch Data" and "Generate Report" as required steps (include relationships) that execute automatically during analysis. "Export to CSV" is a conditional extension that triggers when export is requested.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "booking_reservation": {
        "json": """{
  "actors": [{"name": "Guest"}, {"name": "Hotel System"}, {"name": "Payment Service"}],
  "use_cases": [
    {"name": "Book Room", "description": "Guest books a room"},
    {"name": "Check Availability", "description": "Verify room availability"},
    {"name": "Process Payment", "description": "Handle payment"},
    {"name": "Send Confirmation", "description": "Email booking confirmation"}
  ],
  "relationships": [
    {"type": "association", "from": "Guest", "to": "Book Room"},
    {"type": "include", "from": "Book Room", "to": "Check Availability"},
    {"type": "include", "from": "Book Room", "to": "Process Payment"},
    {"type": "extend", "from": "Book Room", "to": "Send Confirmation"}
  ],
  "overall_description": "Hotel booking system with mandatory availability check and payment, plus optional email confirmation."
}""",
        "instruction": """Definition: In this task, implement the "Book Room" workflow where Guest makes reservations with mandatory availability verification and payment processing.
Emphasis & Caution: You MUST enforce "Check Availability" and "Process Payment" as required steps (include relationships) before confirming booking. "Send Confirmation" is a conditional extension triggered upon successful booking.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "file_management": {
        "json": """{
  "actors": [{"name": "User"}, {"name": "Storage System"}, {"name": "Backup Service"}],
  "use_cases": [
    {"name": "Upload File", "description": "User uploads a file"},
    {"name": "Scan for Viruses", "description": "Check file safety"},
    {"name": "Store File", "description": "Save to storage"},
    {"name": "Create Backup", "description": "Backup the file"}
  ],
  "relationships": [
    {"type": "association", "from": "User", "to": "Upload File"},
    {"type": "include", "from": "Upload File", "to": "Scan for Viruses"},
    {"type": "include", "from": "Upload File", "to": "Store File"},
    {"type": "extend", "from": "Upload File", "to": "Create Backup"}
  ],
  "overall_description": "File upload system with mandatory virus scanning and storage, plus optional backup creation."
}""",
        "instruction": """Definition: In this task, implement the "Upload File" workflow where User uploads files with mandatory virus scanning and storage.
Emphasis & Caution: You MUST enforce "Scan for Viruses" and "Store File" as required steps (include relationships) during upload. "Create Backup" is a conditional extension that triggers based on file importance or user settings.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "notification_system": {
        "json": """{
  "actors": [{"name": "System"}, {"name": "User"}, {"name": "Email Service"}],
  "use_cases": [
    {"name": "Send Notification", "description": "System sends notification"},
    {"name": "Format Message", "description": "Format notification content"},
    {"name": "Deliver to User", "description": "Send to user device"},
    {"name": "Log Activity", "description": "Record notification history"}
  ],
  "relationships": [
    {"type": "association", "from": "System", "to": "Send Notification"},
    {"type": "include", "from": "Send Notification", "to": "Format Message"},
    {"type": "include", "from": "Send Notification", "to": "Deliver to User"},
    {"type": "extend", "from": "Send Notification", "to": "Log Activity"}
  ],
  "overall_description": "Notification system with mandatory message formatting and delivery, plus optional activity logging."
}""",
        "instruction": """Definition: In this task, implement the "Send Notification" workflow where System sends notifications with mandatory message formatting and delivery.
Emphasis & Caution: You MUST enforce "Format Message" and "Deliver to User" as required steps (include relationships) before completing notification. "Log Activity" is a conditional extension that triggers when logging is enabled.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "access_control": {
        "json": """{
  "actors": [{"name": "User"}, {"name": "Admin"}, {"name": "Access Control System"}],
  "use_cases": [
    {"name": "Request Access", "description": "User requests resource access"},
    {"name": "Verify Identity", "description": "Authenticate user"},
    {"name": "Check Permissions", "description": "Verify user permissions"},
    {"name": "Log Access", "description": "Record access attempt"}
  ],
  "relationships": [
    {"type": "association", "from": "User", "to": "Request Access"},
    {"type": "include", "from": "Request Access", "to": "Verify Identity"},
    {"type": "include", "from": "Request Access", "to": "Check Permissions"},
    {"type": "extend", "from": "Request Access", "to": "Log Access"}
  ],
  "overall_description": "Access control system with mandatory identity verification and permission checking, plus optional access logging."
}""",
        "instruction": """Definition: In this task, implement the "Request Access" workflow where User requests resource access with mandatory identity verification and permission checks.
Emphasis & Caution: You MUST enforce "Verify Identity" and "Check Permissions" as required steps (include relationships) before granting access. "Log Access" is a conditional extension that triggers when audit logging is enabled.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    }
}


# ==================== System Prompt (统一使用英文版本) ====================
SYSTEM_PROMPT = """You are a software architecture and crowdsourcing task design expert. Based on the input UML Use Case Diagram structured data (JSON format), write an English task instruction for crowdsourcing workers.

Core Principles:
1. Data-Driven: Actor names and Use Case names in the instruction must strictly reference the original names from JSON source data. Do not omit, abbreviate, or rewrite.
2. Logic Priority, Visuals Secondary: Completely ignore visual layout information like position (e.g., top_left) in input data. Focus on parsing business logic in relationships.
3. Relationship Semantics Translation:
   - include -> Translate to "Mandatory step" or "Required prerequisite"
   - extend -> Translate to "Conditional flow" or "Optional"
   - association -> Translate to "Interaction" or "Access"
4. Structured Format: Strictly follow the three-part format defined below.

Output Format Requirements:
- Definition: Use a clear imperative sentence to describe the core system objective. Must start with "In this task,".
- Emphasis & Caution: Highlight mandatory flows (include) and conditional extension flows (extend). Use "-" if none.
- Things to Avoid: List prohibited operations (e.g., focusing on node positions, implementing UI styles). Use "-" if nothing specific.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- Definition must start with "In this task," and explicitly list actors and use cases from JSON data
- Translate relationship types (include/extend/association) to business logic terms
- Keep all sections concise
- Output ONLY these three lines, nothing else

Reference Example:
{example}

Please generate instructions for the following {count} UML use case diagram(s). Strictly follow the format below and do not add extra explanations:

{uml_data}

"""


# ==================== 修复工具类 ====================
class UMLBatchRepairer:
    """UML批次修复器 - 继承稳定功能并新增完整性检查"""

    def __init__(self):
        self.driver = None
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

            user_data_dir = os.path.join(os.getcwd(), 'chrome_user_data_uml_repair')
            if not os.path.exists(user_data_dir):
                os.makedirs(user_data_dir)
            options.add_argument(f'--user-data-dir={user_data_dir}')

            options.add_argument('--disable-extensions')
            options.add_argument('--remote-debugging-port=9224')
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
        """获取当前页面的回复数量"""
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
        """核心检测方法:通过内容变化判断是否还在生成"""
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
                    if elements:
                        last_element = elements[-1]
                        content_before = last_element.text.strip()
                        time.sleep(1)
                        content_after = last_element.text.strip()

                        if content_before != content_after:
                            return True
                        return False
                except:
                    continue
            return False
        except:
            return False

    def start_new_chat(self):
        """开启新对话"""
        print(f"\n{'='*50}")
        print("  🔄 开启新对话...")
        print(f"{'='*50}")

        try:
            new_chat_selectors = [
                "button[aria-label*='New chat']",
                "button:contains('New chat')",
                "a[href='/']",
                "button.new-chat",
            ]

            for selector in new_chat_selectors:
                try:
                    button = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    button.click()
                    time.sleep(3)
                    print("  ✓ 新对话已开启")
                    self.response_count_before_send = 0
                    return
                except:
                    continue

            print("  ⚠ 未找到新对话按钮,尝试刷新页面")
            self.driver.refresh()
            time.sleep(5)
            print("  ✓ 页面已刷新")
            self.response_count_before_send = 0

        except Exception as e:
            print(f"  ⚠ 开启新对话失败: {e}")
            print("  尝试刷新页面...")
            self.driver.refresh()
            time.sleep(5)
            self.response_count_before_send = 0

    def send_prompt(self, prompt_text):
        """发送prompt"""
        try:
            self.response_count_before_send = self.get_current_response_count()

            input_box = self.find_input_box()
            input_box.clear()
            time.sleep(0.5)

            input_box.send_keys(prompt_text)
            time.sleep(1)

            button = self.find_submit_button()
            if button:
                button.click()
            else:
                input_box.send_keys(Keys.RETURN)

            time.sleep(2)
            return True

        except Exception as e:
            print(f"  ✗ 发送失败: {e}")
            return False

    def wait_for_response_complete(self):
        """等待响应完成"""
        print("  ⏳ 等待GPT响应...")

        time.sleep(3)

        timeout = WAIT_NEW_RESPONSE_TIMEOUT
        start_time = time.time()
        stable_count = 0

        while (time.time() - start_time) < timeout:
            current_count = self.get_current_response_count()

            if current_count > self.response_count_before_send:
                if not self.check_response_still_updating():
                    stable_count += 1
                    if stable_count >= CONTENT_STABLE_CHECKS:
                        print(f"  ✓ 响应完成 (耗时: {int(time.time() - start_time)}秒)")
                        return True
                else:
                    stable_count = 0

            time.sleep(1)

        print(f"  ✗ 等待超时 ({timeout}秒)")
        return False

    def extract_response(self):
        """提取最新的GPT响应"""
        try:
            response_selectors = [
                "div[class*='markdown']",
                "div[data-message-author-role='assistant']",
                "div[class*='message']"
            ]

            for selector in response_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        return elements[-1].text.strip()
                except:
                    continue

            return ""
        except:
            return ""

    def parse_uml_instruction(self, response_text):
        """
        解析UML指令 - 三段式格式
        适配【图像N】格式和普通格式
        返回: instruction字符串 或 None
        """
        # 尝试匹配【图像1】格式
        pattern = r'【图像\d+】\s*\n(.*?)(?=【图像\d+】|$)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        if len(matches) == 1:
            return matches[0].strip()

        # 备用解析方法：按 Definition: 分割
        parts = response_text.split('Definition:')
        for part in parts[1:]:
            if 'Emphasis & Caution:' in part and 'Things to Avoid:' in part:
                return 'Definition:' + part.strip()

        # 最后尝试：直接提取三段式
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
            return f"{definition}\n{emphasis}\n{avoid}"
        else:
            return None

    def extract_domain_from_header(self, header: str) -> str:
        """
        从Header列提取领域名称

        Args:
            header: 图片名（去掉文件扩展名）

        Returns:
            str: 领域名称，如果无法识别则返回"unknown"

        Examples:
            "ecommerce_simple_001" -> "ecommerce"
            "authentication_medium_045" -> "authentication"
            "social_interaction_complex_120" -> "social_interaction"
        """
        header = header.lower()

        # 已知的10个领域列表
        known_domains = [
            "ecommerce", "authentication", "content_management",
            "social_interaction", "customer_service", "data_analysis",
            "permission_management", "notification_system",
            "file_management", "booking_system"
        ]

        # 优先匹配多单词领域（避免误匹配）
        for domain in sorted(known_domains, key=len, reverse=True):
            if domain in header:
                return domain

        return "unknown"

    def get_example_for_domain(self, domain: str) -> str:
        """
        根据领域获取对应的Few-shot示例

        Args:
            domain: 领域名称

        Returns:
            str: 格式化的示例文本
        """
        if domain not in DOMAIN_EXAMPLES:
            # 如果领域未知，使用authentication作为默认示例
            domain = "authentication"

        example_data = DOMAIN_EXAMPLES[domain]
        example_text = f"{example_data['json']}\n\nOutput Instruction:\n{example_data['instruction']}"

        return example_text

    def clean_json_data(self, json_str):
        """
        UML专用：移除position等无关视觉字段
        保留所有业务逻辑相关字段
        """
        try:
            data = json.loads(json_str)

            # 移除actors中的position字段
            if 'actors' in data and isinstance(data['actors'], list):
                for actor in data['actors']:
                    if 'position' in actor:
                        del actor['position']

            # 保留所有其他字段：use_cases, relationships, system_boundary, overall_description
            # 这些都是业务逻辑相关的有效信息

            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ⚠ JSON清洗失败: {e}")
            return json_str  # 返回原始数据

    def validate_instruction_format(self, instruction):
        """
        验证指令格式完整性
        返回: (is_valid, errors)
        """
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
                # 检查Definition是否以"In this task,"开头
                content = line[len('Definition:'):].strip()
                if not content.lower().startswith('in this task'):
                    errors.append("Definition未以'In this task'开头")
                # 检查是否以句号结尾
                if not content.endswith('.'):
                    errors.append("Definition缺少结尾句号")

            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                has_emphasis = True
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                # 检查是否以句号结尾(如果不是"-")
                if content and content != '-' and not content.endswith('.'):
                    errors.append("Emphasis & Caution缺少结尾句号")

            elif line.startswith('Things to Avoid:'):
                has_avoid = True
                content = line[len('Things to Avoid:'):].strip()
                # 检查是否以句号结尾(如果不是"-")
                if content and content != '-' and not content.endswith('.'):
                    errors.append("Things to Avoid缺少结尾句号")

        if not has_definition:
            errors.append("缺少Definition部分")
        if not has_emphasis:
            errors.append("缺少Emphasis & Caution部分")
        if not has_avoid:
            errors.append("缺少Things to Avoid部分")

        is_valid = (has_definition and has_emphasis and has_avoid and len(errors) == 0)
        return is_valid, errors

    def detect_error_batches(self, df):
        """
        检测需要修复的批次
        包括:
        1. 包含"ERROR: 生成失败"的行
        2. 格式不完整的行(三段式检查、句号检查)
        """
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
                header = df.loc[idx, 'Header']
                row_num = idx + 1

                # 检查1: ERROR标记
                if 'ERROR' in instruction.upper() and '生成失败' in instruction:
                    batch_has_error = True
                    error_msg = "含有ERROR标记"
                    batch_error_details.append((row_num, header, error_msg))
                    continue

                # 检查2: 空指令
                if not instruction or instruction.strip() == '' or instruction == 'nan':
                    batch_has_error = True
                    error_msg = "指令为空"
                    batch_error_details.append((row_num, header, error_msg))
                    continue

                # 检查3: 格式完整性
                is_valid, format_errors = self.validate_instruction_format(instruction)
                if not is_valid:
                    batch_has_error = True
                    error_msg = "; ".join(format_errors)
                    batch_error_details.append((row_num, header, error_msg))

            if batch_has_error:
                error_count += len(batch_error_details)
                self.error_details.extend(batch_error_details)

                # 获取该批次的数据
                batch_data = []
                for idx in range(i, batch_end):
                    header = df.loc[idx, 'Header']
                    description = df.loc[idx, 'Description']
                    batch_data.append((header, description))

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

        for row_num, header, error_msg in self.error_details:
            print(f"\n第 {row_num} 条:")
            print(f"  Header: {header[:50]}...")
            print(f"  错误: {error_msg}")

        print("\n" + "="*60)

    def process_batch(self, batch_data, start_idx, batch_num, max_retries=3):
        """
        处理单个批次(使用UML专用Prompt模板)
        batch_data: [(header, description), ...]
        """
        print(f"\n{'='*60}")
        print(f"批次 #{batch_num} | 数据范围: {start_idx + 1}-{start_idx + len(batch_data)}")
        print(f"{'='*60}")

        # 提取领域并选择示例
        first_header = batch_data[0][0] if batch_data else ""
        domain = self.extract_domain_from_header(first_header)
        example_text = self.get_example_for_domain(domain)

        print(f"  🏷️ 识别领域: {domain}")
        print(f"  📝 使用示例: {domain} 领域\n")

        # 构建UML数据文本（清洗后）
        data_text = ""
        for i, (header, description) in enumerate(batch_data, 1):
            # 清洗JSON数据（移除position等无关字段）
            cleaned_json = self.clean_json_data(description)
            data_text += f"{i}. [UML Diagram: {header}]\n{cleaned_json}\n\n"

        prompt = SYSTEM_PROMPT.format(
            example=example_text,
            count=len(batch_data),
            uml_data=data_text
        )

        # 重试循环
        for retry_count in range(max_retries):
            if retry_count > 0:
                print(f"\n🔄 检测到生成错误,正在重试 ({retry_count}/{max_retries - 1})...")
                time.sleep(3)

            if not self.send_prompt(prompt):
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}",
                        'error': '发送失败'
                    })
                    return [None]

            if not self.wait_for_response_complete():
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}",
                        'error': '等待超时'
                    })
                    return [None]

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
                        'range': f"{start_idx + 1}",
                        'error': f'生成错误(重试{max_retries}次后失败)'
                    })
                    return [None]
            else:
                print(f"  ✓ 响应正常,准备解析")
                break

        # 解析指令
        instruction = self.parse_uml_instruction(response)

        if instruction:
            # 验证格式
            is_valid, errors = self.validate_instruction_format(instruction)
            if is_valid:
                print(f"  ✓ 指令格式验证通过")
                return [instruction]
            else:
                print(f"  ⚠ 指令格式验证失败: {errors}")
                return [instruction]  # 仍然返回，后续可能需要手动修复
        else:
            print(f"  ✗ 解析失败")
            return [None]

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

        # 检测需要修复的批次
        error_batches = self.detect_error_batches(df)

        if not error_batches:
            print("\n✓ 未发现需要修复的错误数据")
            # 仍然打印详细报告(如果有的话)
            self.print_error_report()
            return 0

        # 打印详细错误报告
        self.print_error_report()

        # 询问是否继续
        print(f"\n⚠️ 发现 {len(error_batches)} 个错误批次,共约 {len(error_batches) * BATCH_SIZE} 条数据需要修复")
        user_input = input("是否继续修复? (y/n): ").strip().lower()
        if user_input != 'y':
            print("❌ 用户取消修复")
            return 0

        # 开启新对话
        self.start_new_chat()

        # 逐批次修复
        repaired_batches = 0
        for batch_num, start_idx, batch_data in error_batches:
            # 处理批次
            instructions = self.process_batch(batch_data, start_idx, batch_num)

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
            print(f"  ✓ 已保存进度: {start_idx + len(batch_data)}/{len(df)}")

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
        print(f"{'UML批次完整性修复系统':^60}")
        print(f"{'=' * 60}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"批次大小: {BATCH_SIZE} 条/批")
        print(f"检测功能: ERROR标记 + 三段式格式 + 句号检查")
        print(f"目标文件: {CSV_FILE}")
        print(f"{'=' * 60}\n")

        try:
            self.init_driver()

            # 处理目标文件
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
    repairer = UMLBatchRepairer()
    repairer.run()