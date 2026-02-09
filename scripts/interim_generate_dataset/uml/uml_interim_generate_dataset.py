"""
众包指令自动生成脚本 - UML用例图版本
基于UML用例图JSON数据批量生成众包业务逻辑实现指令

优化要点:
1. 专门处理UML用例图结构化数据
2. 数据清洗：移除无关position字段
3. Few-shot学习：包含高质量UML示例
4. 单条处理高质量：BATCH_SIZE=1
5. ✨ 增强生成检测稳定性（支持长响应+瞬间生成）
6. ✨ 修复刷新计数bug（每条后自动刷新）
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
import json

# ==================== 配置参数 ====================
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DATASET_PATH = r"dataset/uml"
GPT_URL = "https://sass-node1.chatshare.biz/"

# ✨ 修改：单个CSV文件
CSV_FILE = "uml_dataset_qwen3_v2.csv"

# ✨ 修改：优化批次参数
BATCH_SIZE = 1  # 每批1条，质量优先
REFRESH_INTERVAL = 1  # 每1条开启新对话（每批都刷新）
CHECK_INTERVAL = 100
TEST_MODE_LIMIT = 10  # 测试模式：每个领域随机1条，总共10条

# ✨ 新增：响应等待时间配置
WAIT_NEW_RESPONSE_TIMEOUT = 60  # 等待新回复最多60秒（应对长响应）
CONTENT_STABLE_CHECKS = 3  # 内容稳定性检查次数

# ==================== 10个领域的优质示例库 ====================
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
  "overall_description": "Data analysis system with mandatory data fetching and report generation, plus optional CSV export functionality."
}""",
        "instruction": """Definition: In this task, implement the "Run Analysis" workflow where Analyst executes data analysis with mandatory data fetching and report generation.
Emphasis & Caution: You MUST enforce "Fetch Data" and "Generate Report" as required steps (include relationships) that execute during analysis. "Export to CSV" is a conditional extension that triggers when the analyst requests data export.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "permission_management": {
        "json": """{
  "actors": [{"name": "Admin"}, {"name": "User"}, {"name": "Audit System"}],
  "use_cases": [
    {"name": "Assign Role", "description": "Assign role to user"},
    {"name": "Validate Permissions", "description": "Check role validity"},
    {"name": "Update Access Rights", "description": "Modify user permissions"},
    {"name": "Log Changes", "description": "Record permission changes"}
  ],
  "relationships": [
    {"type": "association", "from": "Admin", "to": "Assign Role"},
    {"type": "include", "from": "Assign Role", "to": "Validate Permissions"},
    {"type": "include", "from": "Assign Role", "to": "Update Access Rights"},
    {"type": "extend", "from": "Assign Role", "to": "Log Changes"}
  ],
  "overall_description": "Permission management system with mandatory permission validation and access rights updates, plus optional audit logging."
}""",
        "instruction": """Definition: In this task, implement the "Assign Role" workflow where Admin assigns roles to users with mandatory permission validation and access rights updates.
Emphasis & Caution: You MUST enforce "Validate Permissions" and "Update Access Rights" as required steps (include relationships) that execute during role assignment. "Log Changes" is a conditional extension that triggers when audit logging is enabled.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "notification_system": {
        "json": """{
  "actors": [{"name": "System"}, {"name": "User"}, {"name": "Email Service"}, {"name": "SMS Gateway"}],
  "use_cases": [
    {"name": "Send Notification", "description": "Trigger notification"},
    {"name": "Check Preferences", "description": "Verify user notification settings"},
    {"name": "Send Email", "description": "Send email notification"},
    {"name": "Send SMS", "description": "Send SMS notification"}
  ],
  "relationships": [
    {"type": "association", "from": "System", "to": "Send Notification"},
    {"type": "include", "from": "Send Notification", "to": "Check Preferences"},
    {"type": "extend", "from": "Send Notification", "to": "Send Email"},
    {"type": "extend", "from": "Send Notification", "to": "Send SMS"}
  ],
  "overall_description": "Notification system with mandatory preference checking and optional email or SMS delivery based on user settings."
}""",
        "instruction": """Definition: In this task, implement the "Send Notification" workflow where System triggers notifications with mandatory preference checking.
Emphasis & Caution: You MUST enforce "Check Preferences" as a required step (include relationship) that executes before sending notifications. "Send Email" and "Send SMS" are conditional extensions that trigger based on user notification preferences.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "file_management": {
        "json": """{
  "actors": [{"name": "User"}, {"name": "Storage System"}, {"name": "Virus Scanner"}],
  "use_cases": [
    {"name": "Upload File", "description": "User uploads a file"},
    {"name": "Scan for Viruses", "description": "Check file for malware"},
    {"name": "Store File", "description": "Save file to storage"},
    {"name": "Generate Preview", "description": "Create file thumbnail"}
  ],
  "relationships": [
    {"type": "association", "from": "User", "to": "Upload File"},
    {"type": "include", "from": "Upload File", "to": "Scan for Viruses"},
    {"type": "include", "from": "Upload File", "to": "Store File"},
    {"type": "extend", "from": "Upload File", "to": "Generate Preview"}
  ],
  "overall_description": "File upload system with mandatory virus scanning and storage, plus optional preview generation for supported file types."
}""",
        "instruction": """Definition: In this task, implement the "Upload File" workflow where User uploads files with mandatory virus scanning and storage operations.
Emphasis & Caution: You MUST enforce "Scan for Viruses" and "Store File" as required steps (include relationships) that execute during file upload. "Generate Preview" is a conditional extension that triggers for supported file types (images, documents).
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "booking_system": {
        "json": """{
  "actors": [{"name": "Customer"}, {"name": "Calendar System"}, {"name": "Payment Gateway"}],
  "use_cases": [
    {"name": "Book Appointment", "description": "Customer books a time slot"},
    {"name": "Check Availability", "description": "Verify slot availability"},
    {"name": "Process Payment", "description": "Handle booking payment"},
    {"name": "Send Reminder", "description": "Send appointment reminder"}
  ],
  "relationships": [
    {"type": "association", "from": "Customer", "to": "Book Appointment"},
    {"type": "include", "from": "Book Appointment", "to": "Check Availability"},
    {"type": "include", "from": "Book Appointment", "to": "Process Payment"},
    {"type": "extend", "from": "Book Appointment", "to": "Send Reminder"}
  ],
  "overall_description": "Appointment booking system with mandatory availability checking and payment processing, plus optional reminder notifications."
}""",
        "instruction": """Definition: In this task, implement the "Book Appointment" workflow where Customer books time slots with mandatory availability checking and payment processing.
Emphasis & Caution: You MUST enforce "Check Availability" and "Process Payment" as required steps (include relationships) that execute during booking. "Send Reminder" is a conditional extension that triggers when reminder notifications are enabled.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    }
}

# ✨ System Prompt (统一使用英文版本，参考uml_template.py)
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

Output format (3 lines only):

Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...
"""

# ==================== 工具函数 ====================
class GPTAutomator:
    def __init__(self, test_mode=True):
        self.test_mode = test_mode
        self.driver = None
        self.current_tab = None
        self.processed_count = 0
        self.error_log = []

        # 【优化】缓存成功的选择器
        self.cached_input_selector = None
        self.cached_button_selector = None

        # 【新增】记录发送前的回复数量，避免检测到旧回复
        self.response_count_before_send = 0

        # ✨ 【新增】批次计数器（修复刷新bug）
        self.batches_since_refresh = 0

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
        ✨ UML专用：移除position等无关视觉字段
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

    def find_input_box(self, debug=False):
        """定位输入框 - 优化版,使用缓存"""
        if debug:
            print("🔍 定位输入框...")

        # 【优化】如果有缓存的选择器,优先使用
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
                    # 缓存失效,清除
                    self.cached_input_selector = None
            except:
                self.cached_input_selector = None

        # 【优化】调整选择器优先级,把成功率高的放前面
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
                    # 【优化】缓存成功的选择器
                    self.cached_input_selector = selector
                    if debug:
                        print(f"  ✓ 成功: {selector}")
                    return element

            except:
                continue

        raise NoSuchElementException("无法找到输入框")

    def find_submit_button(self):
        """定位提交按钮 - 优化版,使用缓存"""
        # 【优化】如果有缓存的选择器,优先使用
        if self.cached_button_selector:
            try:
                button = self.driver.find_element(By.CSS_SELECTOR, self.cached_button_selector)
                if button.is_displayed() and button.is_enabled():
                    return button
                else:
                    self.cached_button_selector = None
            except:
                self.cached_button_selector = None

        # 【优化】按成功率排序
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
                        # 【优化】缓存成功的选择器
                        self.cached_button_selector = selector
                        return button
            except:
                continue

        return None

    def get_current_response_count(self):
        """
        ✨ 精确修复：基于实际DOM结构获取assistant回复数量
        使用data-message-author-role="assistant"作为准确标记
        """
        try:
            # 🎯 最精确的选择器（基于您提供的HTML结构）
            response_selectors = [
                # 方法1：直接定位assistant消息（最可靠）
                "div[data-message-author-role='assistant']",
                # 方法2：通过article容器定位
                "article[data-turn='assistant']",
                # 方法3：备用 - markdown容器（但需要排除用户消息）
                "article[data-testid*='conversation-turn'] div.markdown.prose",
            ]

            for selector in response_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)

                    if selector == "article[data-testid*='conversation-turn'] div.markdown.prose":
                        # 对于markdown选择器，需要确认是assistant的
                        valid_count = 0
                        for elem in elements:
                            try:
                                # 检查父article是否是assistant
                                parent_article = self.driver.execute_script(
                                    "return arguments[0].closest('article')",
                                    elem
                                )
                                if parent_article:
                                    turn_type = parent_article.get_attribute('data-turn')
                                    if turn_type == 'assistant':
                                        valid_count += 1
                            except:
                                continue
                        if valid_count > 0:
                            return valid_count
                    else:
                        # 对于前两个选择器，直接返回数量
                        if elements and len(elements) > 0:
                            return len(elements)
                except:
                    continue
            return 0
        except:
            return 0

    def check_response_still_updating(self):
        """
        ✨ 精确修复：基于实际DOM检测内容是否还在生成
        """
        try:
            # 🎯 使用最精确的assistant选择器
            selector = "div[data-message-author-role='assistant']"

            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            current_count = len(elements)

            # 确保检测的是新生成的回复
            if current_count <= self.response_count_before_send:
                return True  # 还没有新回复

            if not elements:
                return True

            # 获取最后一条assistant回复
            last_response = elements[-1]
            first_text = last_response.text
            first_len = len(first_text)

            time.sleep(0.8)  # 等待0.8秒检测变化

            # 重新获取
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            if len(elements) > 0:
                last_response = elements[-1]
                second_text = last_response.text
                second_len = len(second_text)

                # 内容增加 = 还在生成
                if second_len > first_len:
                    return True
                return False

            return False
        except Exception as e:
            print(f"检测更新异常: {e}")
            return False

    def wait_for_response_complete(self, timeout=300):
        """
        ✨✨ 终极修复：解决验证死循环和卡死问题

        核心改进：
        1. 限制连续验证失败次数（避免死循环）
        2. 放宽验证条件（提高兼容性）
        3. 增加超时逃生机制
        """
        print("  等待生成...", end='', flush=True)
        start_time = time.time()
        last_progress_time = start_time

        # ✨ 阶段1：等待新回复出现
        print(" [等待响应]", end='', flush=True)
        response_appeared = False

        # 🆕 防死循环：限制连续验证失败次数
        consecutive_validation_failures = 0
        MAX_VALIDATION_FAILURES = 50  # 连续失败后强制接受

        check_count = 0
        while time.time() - start_time < WAIT_NEW_RESPONSE_TIMEOUT:
            try:
                current_count = self.get_current_response_count()

                # 🔧 检测到数量增加
                if current_count > self.response_count_before_send:
                    print(f" [检测到可能的新回复,验证中]", end='', flush=True)
                    time.sleep(2)  # 等待DOM稳定

                    # 再次确认数量
                    recheck_count = self.get_current_response_count()

                    if recheck_count > self.response_count_before_send:
                        # 🔧 验证内容
                        if self._validate_new_response():
                            elapsed = int(time.time() - start_time)
                            response_appeared = True
                            print(f" ✓ [新回复已确认,耗时{elapsed}s]", end='', flush=True)
                            break
                        else:
                            # 🆕 关键修复：累计验证失败次数
                            consecutive_validation_failures += 1
                            print(f" [内容验证失败{consecutive_validation_failures}/{MAX_VALIDATION_FAILURES}]", end='',
                                  flush=True)

                            # 🚨 如果连续失败太多次，强制接受（避免死循环）
                            if consecutive_validation_failures >= MAX_VALIDATION_FAILURES:
                                elapsed = int(time.time() - start_time)
                                print(f" ⚠️ [验证失败但强制接受,耗时{elapsed}s]", end='', flush=True)
                                response_appeared = True
                                break

                            # 继续等待，但增加等待时间
                            time.sleep(2)
                    else:
                        print(f" [数量未稳定,继续等待]", end='', flush=True)
                        consecutive_validation_failures = 0  # 重置计数器
                        time.sleep(1)
                else:
                    # 数量还没增加，重置失败计数器
                    consecutive_validation_failures = 0

                # 每2秒打印进度
                check_count += 1
                if check_count % 4 == 0:
                    elapsed = int(time.time() - start_time)
                    print(f"[{elapsed}s]", end='', flush=True)

            except Exception as e:
                print(f"![{str(e)[:20]}]", end='', flush=True)

            time.sleep(0.5)

        if not response_appeared:
            print(f" ✗ 等待响应超时({WAIT_NEW_RESPONSE_TIMEOUT}s)")
            return False

        # ✨ 阶段2：等待内容生成完毕
        time.sleep(1)
        print(" [检测完成]", end='', flush=True)

        stable_count = 0
        max_stability_checks = 10

        for check_round in range(max_stability_checks):
            try:
                is_updating = self.check_response_still_updating()

                if is_updating:
                    stable_count = 0
                    print(".", end='', flush=True)
                else:
                    stable_count += 1
                    if stable_count >= CONTENT_STABLE_CHECKS:
                        print(" ✓ 完成")
                        return True
                    else:
                        print(".", end='', flush=True)

                current_time = time.time()
                if current_time - last_progress_time >= 5:
                    total_elapsed = int(current_time - start_time)
                    print(f"[{total_elapsed}s]", end='', flush=True)
                    last_progress_time = current_time

                time.sleep(1)

            except Exception as e:
                print(f"⚠ [{str(e)[:20]}]", end='', flush=True)
                time.sleep(1)

        print(" ✓ 完成(达到检查上限)")
        return True

    def extract_response(self):
        """
        ✨ 精确提取：基于data-message-author-role='assistant'提取回复
        """
        try:
            # 🎯 使用最可靠的选择器
            selector = "div[data-message-author-role='assistant']"

            response_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            current_count = len(response_elements)

            # 确保提取的是新生成的回复
            if current_count > self.response_count_before_send:
                # 获取最后一条（最新的回复）
                last_response = response_elements[-1]

                # 尝试获取markdown内容（更干净）
                try:
                    markdown_div = last_response.find_element(
                        By.CSS_SELECTOR,
                        "div.markdown.prose"
                    )
                    response_text = markdown_div.text
                except:
                    # 如果没有markdown容器，直接获取文本
                    response_text = last_response.text

                # 验证内容有效性
                if response_text and len(response_text) > 10:
                    print(f"  ✓ 提取到回复 ({len(response_text)} 字符)")

                    # 验证：检查是否包含预期关键词
                    has_definition = "Definition:" in response_text
                    has_emphasis = "Emphasis" in response_text or "Caution" in response_text
                    has_avoid = "Avoid" in response_text

                    if has_definition or has_emphasis or has_avoid:
                        print(f"  ✓ 内容验证通过（包含指令关键词）")
                    else:
                        print(f"  ⚠ 警告：回复可能不包含预期格式")

                    return response_text
                else:
                    print(f"  ⚠ 提取的内容太短: {len(response_text) if response_text else 0} 字符")

            print(f"  ✗ 无法提取有效回复（当前{current_count}条，发送前{self.response_count_before_send}条）")
            return ""

        except Exception as e:
            print(f"  ✗ 提取回复失败: {e}")
            return ""

    def _validate_new_response(self):
        """
        🆕 放宽验证条件：提高兼容性，减少误判
        """
        try:
            selector = "div[data-message-author-role='assistant']"
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)

            if not elements or len(elements) <= self.response_count_before_send:
                return False

            last_response = elements[-1]

            # 尝试从markdown容器获取文本
            try:
                markdown_div = last_response.find_element(
                    By.CSS_SELECTOR,
                    "div.markdown.prose"
                )
                text = markdown_div.text.strip()
            except:
                text = last_response.text.strip()

            # 🔧 放宽验证1：文本长度（降低到5个字符）
            if len(text) < 5:
                return False

            # 🔧 简化验证2：只排除明显的加载状态
            # 如果只有加载图标且长度很短
            if len(text) <= 3 and text in ["●", "⚫", "🔴", "...", "•"]:
                return False

            # 🔧 放宽验证3：允许更多特殊字符
            # 只要有任何字母、数字、中文字符就算有效
            has_content = any(c.isalnum() or '\u4e00' <= c <= '\u9fff' for c in text)
            if not has_content and len(text) < 20:  # 如果没有字母数字但长度够长也接受
                return False

            return True

        except Exception as e:
            # 🔧 验证异常时，如果有足够的等待时间，倾向于接受
            print(f"[验证异常,接受]", end='', flush=True)
            return True  # 改为True，避免因异常导致死循环

    def parse_instructions(self, response_text, expected_count):
        """
        ✨ 修改：解析LLM回复,提取UML业务逻辑指令
        适配新的格式：【图像N】
        """
        instructions = []

        # 尝试匹配新格式：【图像N】
        pattern = r'【图像\d+】\s*\n(.*?)(?=【图像\d+】|$)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        if len(matches) == expected_count:
            for match in matches:
                instructions.append(match.strip())
        else:
            # 备用解析方法：按 Definition: 分割
            parts = response_text.split('Definition:')
            for part in parts[1:]:
                if 'Emphasis & Caution:' in part and 'Things to Avoid:' in part:
                    instructions.append('Definition:' + part.strip())

        return instructions

    def send_prompt(self, prompt_text, max_retries=3):
        """
        发送提示词到LLM - 优化版
        【修复】发送前记录当前回复数量
        """
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print(f"\n📤 发送提示词...")
                    # 【关键修复】发送前记录当前回复数量
                    self.response_count_before_send = self.get_current_response_count()
                    print(f"  📊 当前页面已有 {self.response_count_before_send} 条回复")
                else:
                    print(f"  🔄 重试 {attempt}/{max_retries-1}...")

                # 定位输入框
                input_box = self.find_input_box(debug=(attempt == 0))
                if not input_box:
                    if attempt < max_retries - 1:
                        self.driver.refresh()
                        time.sleep(8)
                        continue
                    return False

                # 聚焦并清空
                self.driver.execute_script("arguments[0].focus();", input_box)
                time.sleep(0.3)

                tag_name = input_box.tag_name.lower()
                if tag_name == "textarea":
                    self.driver.execute_script("arguments[0].value = '';", input_box)
                else:
                    self.driver.execute_script("arguments[0].textContent = '';", input_box)

                time.sleep(0.3)

                # 设置文本
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

                # 验证文本设置成功
                current_value = input_box.get_attribute("value") if tag_name == "textarea" else input_box.text
                if not current_value or len(current_value) < 100:
                    if attempt < max_retries - 1:
                        continue
                    return False

                print(f"  ✓ 文本设置成功 ({len(current_value)} 字符)")

                # 发送消息
                button = self.find_submit_button()
                if button:
                    self.driver.execute_script("arguments[0].click();", button)
                    print(f"  ✓ 点击发送按钮")
                else:
                    # 使用Enter键
                    input_box.send_keys(Keys.RETURN)
                    print(f"  ✓ 使用Enter发送")

                time.sleep(2)

                # 验证发送成功
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

    def process_batch(self, uml_data_batch, start_idx):
        """
        ✨ 优化：处理一批UML数据(1条)
        【核心改进】根据Header识别领域，动态选择Few-shot示例
        检测生成错误并自动重试
        【新增】返回是否发生重试的标志
        """
        print(f"\n{'=' * 60}")
        print(f"处理第 {start_idx + 1}-{start_idx + len(uml_data_batch)} 条UML数据")
        print(f"{'=' * 60}")

        # ✨ 核心改进：识别领域并选择示例
        first_header = uml_data_batch[0][0] if uml_data_batch else ""
        domain = self.extract_domain_from_header(first_header)
        example_text = self.get_example_for_domain(domain)

        print(f"  🏷️ 识别领域: {domain}")
        print(f"  📝 使用示例: {domain} 领域\n")

        # ✨ 构建UML数据文本（清洗后）
        data_text = ""
        for i, (header, description) in enumerate(uml_data_batch, 1):
            # 清洗JSON数据（移除position等无关字段）
            cleaned_json = self.clean_json_data(description)
            data_text += f"{i}. [UML Diagram: {header}]\n{cleaned_json}\n\n"

        prompt = SYSTEM_PROMPT.format(
            example=example_text,  # ✨ 使用领域匹配的示例
            count=len(uml_data_batch),
            uml_data=data_text
        )

        # ✨ 最大重试次数
        max_retries = 3
        response = None
        retry_happened = False

        for retry_count in range(max_retries):
            if retry_count > 0:
                retry_happened = True
                print(f"\n🔄 检测到生成错误,正在重试 ({retry_count}/{max_retries - 1})...")
                time.sleep(3)

            # 发送提示词
            if not self.send_prompt(prompt):
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(uml_data_batch)}",
                        'error': '发送失败'
                    })
                    return [None] * len(uml_data_batch), retry_happened

            # 等待响应完成
            if not self.wait_for_response_complete():
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(uml_data_batch)}",
                        'error': '等待超时'
                    })
                    return [None] * len(uml_data_batch), retry_happened

            # 提取响应
            response = self.extract_response()
            print(f"\n响应预览: {response[:200]}...\n")

            # ✨ 核心修改:检测是否为错误响应
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

            # 如果检测到错误响应
            if is_error_response:
                print(f"  ⚠️ 检测到生成错误: {response[:100]}")
                if retry_count < max_retries - 1:
                    print(f"  ↻ 将在3秒后重新发送...")
                    continue
                else:
                    print(f"  ✗ 已达到最大重试次数({max_retries}),放弃本批次")
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(uml_data_batch)}",
                        'error': f'生成错误(重试{max_retries}次后失败)'
                    })
                    return [None] * len(uml_data_batch), retry_happened
            else:
                print(f"  ✓ 响应正常,准备解析")
                break

        # 解析指令
        instructions = self.parse_instructions(response, len(uml_data_batch))

        if len(instructions) != len(uml_data_batch):
            print(f"  ⚠ 警告: 期望{len(uml_data_batch)}条,实际{len(instructions)}条")
            while len(instructions) < len(uml_data_batch):
                instructions.append(None)
            instructions = instructions[:len(uml_data_batch)]

        return instructions, retry_happened

    def start_new_chat(self):
        """
        ✨ 优化版:使用 Ctrl+Shift+O 快捷键开启新对话
        更稳定、高效,避免复杂的按钮查找逻辑
        """
        print("\n>>> 开启新对话...")
        try:
            # 直接使用键盘快捷键 Ctrl+Shift+O
            from selenium.webdriver.common.action_chains import ActionChains

            print("  🔨 发送快捷键 Ctrl+Shift+O...")
            actions = ActionChains(self.driver)
            actions.key_down(Keys.CONTROL).key_down(Keys.SHIFT).send_keys('o').key_up(Keys.SHIFT).key_up(
                Keys.CONTROL).perform()
            print("  ✓ 快捷键已发送")

            # 等待新对话页面加载
            time.sleep(3)

            # 【重要】清除缓存和回复计数
            self.cached_input_selector = None
            self.cached_button_selector = None
            self.response_count_before_send = 0

            # ✨ 【新增】重置批次计数器
            self.batches_since_refresh = 0

            print("  ✓ 新对话已就绪\n")

        except Exception as e:
            print(f"  ✗ 开启新对话失败: {e}")
            print("  ℹ 将继续在当前对话中处理")

    def process_file(self, csv_path):
        """
        ✨✨ 修复：处理UML数据CSV文件
        【关键修复】正确处理刷新计数，避免对话过长
        """
        print(f"\n{'#' * 60}")
        print(f"# 处理文件: {os.path.basename(csv_path)}")
        print(f"{'#' * 60}")

        # 【关键修复1】每个文件开始前强制开启新对话
        self.start_new_chat()

        try:
            # 读取CSV文件
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

        # ✨ 验证列名
        required_columns = ['Header', 'Description', 'Instruction']
        if not all(col in df.columns for col in required_columns):
            print(f"✗ CSV文件缺少必要的列: {required_columns}")
            print(f"  当前列: {df.columns.tolist()}")
            return 0

        # 【修复】明确设置Instruction列为字符串类型，避免FutureWarning
        if 'Instruction' not in df.columns:
            df['Instruction'] = ''
        df['Instruction'] = df['Instruction'].astype(str)
        # 将 'nan' 字符串替换为空字符串
        df.loc[df['Instruction'] == 'nan', 'Instruction'] = ''

        total_rows = len(df)

        if self.test_mode:
            # ✨✨ 新策略：每个领域随机选择1条，总共10条
            print(f"*** 测试模式: 每个领域随机选择1条数据 ***\n")

            # 为每条数据标记领域
            df['domain'] = df['Header'].apply(lambda h: self.extract_domain_from_header(h))

            # 统计每个领域的数量
            domain_counts = df['domain'].value_counts()
            print("领域分布：")
            for domain, count in domain_counts.items():
                print(f"  {domain}: {count} 条")

            # 从每个领域随机选择1条
            selected_indices = []
            for domain in domain_counts.index:
                domain_df = df[df['domain'] == domain]
                if len(domain_df) > 0:
                    sampled = domain_df.sample(n=1, random_state=None)
                    selected_indices.extend(sampled.index.tolist())

            # 按索引排序（保持原始顺序）
            selected_indices.sort()

            print(f"\n已选择 {len(selected_indices)} 条数据进行测试：")
            for idx in selected_indices:
                header = df.loc[idx, 'Header']
                domain = df.loc[idx, 'domain']
                print(f"  - {header} (领域: {domain})")
            print()

            # 创建测试子集
            df = df.loc[selected_indices].reset_index(drop=True)
            total_rows = len(df)

        print(f"总计需处理: {total_rows} 条UML数据\n")
        print(f"刷新策略: 每{REFRESH_INTERVAL}条数据（{REFRESH_INTERVAL//BATCH_SIZE}批）或发生重试后刷新对话\n")

        for i in range(0, total_rows, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_rows)

            # ✨✨✨ 【核心修复】在批次开始前检查是否需要刷新
            # 这样可以在资源累积过多之前就刷新对话
            if i > 0 and self.batches_since_refresh >= (REFRESH_INTERVAL // BATCH_SIZE):
                print(f"\n  🔄 预防性刷新 - 已处理{self.batches_since_refresh}批({self.batches_since_refresh * BATCH_SIZE}条数据)")
                print(f"  ℹ️ 当前进度: {i}/{total_rows}")
                self.start_new_chat()

            # ✨ 修改：提取 Header 和 Description
            batch_data = []
            for idx in range(i, batch_end):
                header = df.loc[idx, 'Header']
                description = df.loc[idx, 'Description']
                batch_data.append((header, description))

            # 【关键】获取批处理结果和重试标志
            instructions, retry_happened = self.process_batch(batch_data, i)

            for j, instruction in enumerate(instructions):
                if instruction:
                    df.at[i + j, 'Instruction'] = instruction
                else:
                    df.at[i + j, 'Instruction'] = "ERROR: 生成失败"

            self.processed_count += len(instructions)
            self.batches_since_refresh += 1  # 【修复】每批都计数

            # ✨✨ 【补充】发生重试后也立即刷新
            if retry_happened and (i + BATCH_SIZE) < total_rows:
                print(f"\n  🔄 检测到重试 - 立即刷新对话避免上下文混乱")
                print(f"  ℹ️ 当前进度: {i + BATCH_SIZE}/{total_rows}")
                self.start_new_chat()

            # 定期保存进度
            if (i + BATCH_SIZE) % 50 == 0:
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"\n  💾 已保存进度: {i + BATCH_SIZE}/{total_rows}\n")

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 文件处理完成: {os.path.basename(csv_path)}")
        print(f"  已处理 {total_rows} 条UML数据\n")

        return total_rows

    def run(self):
        """主运行函数"""
        start_time = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"{'批量UML业务逻辑指令生成系统':^60}")
        print(f"{'=' * 60}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模式: {'测试模式 (每领域1条,共10条)' if self.test_mode else '完整模式'}")
        print(f"批次大小: {BATCH_SIZE} 条/批")
        print(f"刷新间隔: {REFRESH_INTERVAL} 条 ({REFRESH_INTERVAL//BATCH_SIZE} 批)")
        print(f"响应超时: {WAIT_NEW_RESPONSE_TIMEOUT} 秒")
        print(f"{'=' * 60}\n")

        try:
            self.init_driver()

            # ✨ 处理单个CSV文件
            csv_path = os.path.join(DATASET_PATH, CSV_FILE)
            if os.path.exists(csv_path):
                total_processed = self.process_file(csv_path)
            else:
                print(f"✗ 文件不存在: {csv_path}")
                total_processed = 0

            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n{'=' * 60}")
            print(f"{'处理完成':^60}")
            print(f"{'=' * 60}")
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"耗时: {duration}")
            print(f"总计处理: {total_processed} 条UML数据")
            print(f"成功: {total_processed - len(self.error_log)} 条")
            print(f"失败: {len(self.error_log)} 条")

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
    print("\n" + "="*60)
    print("请选择运行模式:")
    print(f"  1. 测试模式 (每个领域随机1条,总共10条)")
    print("  2. 完整模式 (处理所有数据)")
    print("="*60)

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

    automator = GPTAutomator(test_mode=test_mode)
    automator.run()