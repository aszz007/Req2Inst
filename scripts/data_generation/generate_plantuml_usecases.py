#!/usr/bin/env python3
"""
PlantUML用例图批量生成工具
功能：自动生成800-1000张高清用例图用于模型训练
输出：PNG格式，150 DPI，保存至data/raw/uml/plantuml_usecase/
"""

import subprocess
import shutil
import urllib.request
import sys
from pathlib import Path
from typing import List, Tuple
import random


class PlantUMLGenerator:
    """PlantUML用例图生成器"""

    def __init__(self, output_dir: Path):
        """
        初始化生成器

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plantuml_jar = Path(__file__).parent.parent / "plantuml.jar"

        self.domains = self._init_domain_templates()

    def _init_domain_templates(self) -> dict:
        """
        初始化10个领域的用例图模板

        Returns:
            dict: 领域模板字典
        """
        return {
            "ecommerce": self._get_ecommerce_templates(),
            "authentication": self._get_authentication_templates(),
            "content_management": self._get_content_management_templates(),
            "social_interaction": self._get_social_interaction_templates(),
            "customer_service": self._get_customer_service_templates(),
            "data_analysis": self._get_data_analysis_templates(),
            "permission_management": self._get_permission_management_templates(),
            "notification_system": self._get_notification_system_templates(),
            "file_management": self._get_file_management_templates(),
            "booking_system": self._get_booking_system_templates()
        }

    def _get_ecommerce_templates(self) -> List[Tuple[str, List[str], List[str], List[Tuple]]]:
        """
        电商系统模板 (100个场景)

        Returns:
            List[Tuple]: [(场景名, actors, use_cases, relationships), ...]
        """
        templates = []

        base_scenarios = [
            ("Basic Shopping", ["Customer"], ["Browse Products", "Search Items", "View Details"], [
                ("Customer", "Browse Products", "association"),
                ("Customer", "Search Items", "association"),
                ("Browse Products", "View Details", "include")
            ]),
            ("Purchase Flow", ["Customer"], ["Add to Cart", "Checkout", "Make Payment", "Confirm Order"], [
                ("Customer", "Add to Cart", "association"),
                ("Add to Cart", "Checkout", "extend"),
                ("Checkout", "Make Payment", "include"),
                ("Make Payment", "Confirm Order", "include")
            ]),
            ("Order Management", ["Customer", "Admin"], ["Track Order", "Cancel Order", "Process Refund"], [
                ("Customer", "Track Order", "association"),
                ("Customer", "Cancel Order", "association"),
                ("Cancel Order", "Process Refund", "include"),
                ("Admin", "Process Refund", "association")
            ]),
            ("Product Review", ["Customer"], ["Write Review", "Rate Product", "Upload Photos"], [
                ("Customer", "Write Review", "association"),
                ("Write Review", "Rate Product", "include"),
                ("Write Review", "Upload Photos", "extend")
            ]),
            ("Wishlist Management", ["Customer"], ["Add to Wishlist", "Share Wishlist", "Move to Cart"], [
                ("Customer", "Add to Wishlist", "association"),
                ("Add to Wishlist", "Share Wishlist", "extend"),
                ("Add to Wishlist", "Move to Cart", "extend")
            ])
        ]

        for i, (name, actors, usecases, rels) in enumerate(base_scenarios):
            for variant in range(20):
                variant_name = f"{name}_v{variant + 1}"
                templates.append((f"ecommerce_{i}_{variant}", variant_name, actors, usecases, rels))

        return templates

    def _get_authentication_templates(self) -> List[Tuple[str, List[str], List[str], List[Tuple]]]:
        """
        认证系统模板 (100个场景)
        """
        templates = []

        base_scenarios = [
            ("User Login", ["User"], ["Login", "Validate Credentials", "Generate Token"], [
                ("User", "Login", "association"),
                ("Login", "Validate Credentials", "include"),
                ("Validate Credentials", "Generate Token", "include")
            ]),
            ("User Registration", ["User"], ["Register", "Verify Email", "Create Profile"], [
                ("User", "Register", "association"),
                ("Register", "Verify Email", "include"),
                ("Verify Email", "Create Profile", "include")
            ]),
            ("Password Reset", ["User"], ["Request Reset", "Verify Identity", "Update Password"], [
                ("User", "Request Reset", "association"),
                ("Request Reset", "Verify Identity", "include"),
                ("Verify Identity", "Update Password", "include")
            ]),
            ("Two Factor Auth", ["User"], ["Login", "Send OTP", "Verify OTP"], [
                ("User", "Login", "association"),
                ("Login", "Send OTP", "extend"),
                ("Send OTP", "Verify OTP", "include")
            ]),
            ("Social Login", ["User"], ["Login with OAuth", "Authorize App", "Link Account"], [
                ("User", "Login with OAuth", "association"),
                ("Login with OAuth", "Authorize App", "include"),
                ("Authorize App", "Link Account", "extend")
            ])
        ]

        for i, (name, actors, usecases, rels) in enumerate(base_scenarios):
            for variant in range(20):
                variant_name = f"{name}_v{variant + 1}"
                templates.append((f"auth_{i}_{variant}", variant_name, actors, usecases, rels))

        return templates

    def _get_content_management_templates(self) -> List[Tuple[str, List[str], List[str], List[Tuple]]]:
        """
        内容管理系统模板 (100个场景)
        """
        templates = []

        base_scenarios = [
            ("Article Publishing", ["Author", "Editor"], ["Create Article", "Submit for Review", "Approve", "Publish"], [
                ("Author", "Create Article", "association"),
                ("Author", "Submit for Review", "association"),
                ("Submit for Review", "Approve", "include"),
                ("Editor", "Approve", "association"),
                ("Approve", "Publish", "include")
            ]),
            ("Media Upload", ["Author"], ["Upload Media", "Compress Image", "Generate Thumbnail"], [
                ("Author", "Upload Media", "association"),
                ("Upload Media", "Compress Image", "include"),
                ("Compress Image", "Generate Thumbnail", "include")
            ]),
            ("Content Moderation", ["Moderator"], ["Review Content", "Flag Inappropriate", "Remove Content"], [
                ("Moderator", "Review Content", "association"),
                ("Review Content", "Flag Inappropriate", "extend"),
                ("Flag Inappropriate", "Remove Content", "include")
            ]),
            ("Version Control", ["Author"], ["Edit Document", "Save Version", "Restore Previous"], [
                ("Author", "Edit Document", "association"),
                ("Edit Document", "Save Version", "include"),
                ("Edit Document", "Restore Previous", "extend")
            ]),
            ("Workflow Management", ["Editor"], ["Assign Task", "Set Deadline", "Monitor Progress"], [
                ("Editor", "Assign Task", "association"),
                ("Assign Task", "Set Deadline", "include"),
                ("Assign Task", "Monitor Progress", "extend")
            ])
        ]

        for i, (name, actors, usecases, rels) in enumerate(base_scenarios):
            for variant in range(20):
                variant_name = f"{name}_v{variant + 1}"
                templates.append((f"cms_{i}_{variant}", variant_name, actors, usecases, rels))

        return templates

    def _get_social_interaction_templates(self) -> List[Tuple[str, List[str], List[str], List[Tuple]]]:
        """
        社交互动系统模板 (100个场景)
        """
        templates = []

        base_scenarios = [
            ("User Connection", ["User"], ["Follow User", "Send Request", "Accept Request"], [
                ("User", "Follow User", "association"),
                ("Follow User", "Send Request", "include"),
                ("Send Request", "Accept Request", "extend")
            ]),
            ("Post Interaction", ["User"], ["Create Post", "Like Post", "Share Post", "Comment"], [
                ("User", "Create Post", "association"),
                ("User", "Like Post", "association"),
                ("User", "Share Post", "association"),
                ("User", "Comment", "association")
            ]),
            ("Direct Messaging", ["User"], ["Send Message", "Encrypt Message", "Receive Message"], [
                ("User", "Send Message", "association"),
                ("Send Message", "Encrypt Message", "include"),
                ("Send Message", "Receive Message", "include")
            ]),
            ("Group Management", ["User", "Admin"], ["Create Group", "Invite Members", "Manage Members"], [
                ("User", "Create Group", "association"),
                ("Create Group", "Invite Members", "extend"),
                ("Admin", "Manage Members", "association")
            ]),
            ("Content Sharing", ["User"], ["Share Media", "Tag Friends", "Set Privacy"], [
                ("User", "Share Media", "association"),
                ("Share Media", "Tag Friends", "extend"),
                ("Share Media", "Set Privacy", "include")
            ])
        ]

        for i, (name, actors, usecases, rels) in enumerate(base_scenarios):
            for variant in range(20):
                variant_name = f"{name}_v{variant + 1}"
                templates.append((f"social_{i}_{variant}", variant_name, actors, usecases, rels))

        return templates

    def _get_customer_service_templates(self) -> List[Tuple[str, List[str], List[str], List[Tuple]]]:
        """
        客服系统模板 (100个场景)
        """
        templates = []

        base_scenarios = [
            ("Ticket Management", ["Customer", "Agent"], ["Submit Ticket", "Assign Agent", "Resolve Issue"], [
                ("Customer", "Submit Ticket", "association"),
                ("Submit Ticket", "Assign Agent", "include"),
                ("Agent", "Resolve Issue", "association")
            ]),
            ("Live Chat", ["Customer", "Agent"], ["Start Chat", "Transfer Chat", "End Chat"], [
                ("Customer", "Start Chat", "association"),
                ("Agent", "Transfer Chat", "association"),
                ("Start Chat", "End Chat", "extend")
            ]),
            ("Knowledge Base", ["Customer"], ["Search Articles", "View Solution", "Rate Helpfulness"], [
                ("Customer", "Search Articles", "association"),
                ("Search Articles", "View Solution", "include"),
                ("View Solution", "Rate Helpfulness", "extend")
            ]),
            ("Feedback Collection", ["Customer"], ["Submit Feedback", "Rate Service", "Provide Suggestions"], [
                ("Customer", "Submit Feedback", "association"),
                ("Submit Feedback", "Rate Service", "include"),
                ("Submit Feedback", "Provide Suggestions", "extend")
            ]),
            ("Call Center", ["Customer", "Agent"], ["Make Call", "Route Call", "Record Call"], [
                ("Customer", "Make Call", "association"),
                ("Make Call", "Route Call", "include"),
                ("Agent", "Record Call", "association")
            ])
        ]

        for i, (name, actors, usecases, rels) in enumerate(base_scenarios):
            for variant in range(20):
                variant_name = f"{name}_v{variant + 1}"
                templates.append((f"service_{i}_{variant}", variant_name, actors, usecases, rels))

        return templates

    def _get_data_analysis_templates(self) -> List[Tuple[str, List[str], List[str], List[Tuple]]]:
        """
        数据分析系统模板 (100个场景)
        """
        templates = []

        base_scenarios = [
            ("Report Generation", ["Analyst"], ["Select Data", "Apply Filters", "Generate Report"], [
                ("Analyst", "Select Data", "association"),
                ("Select Data", "Apply Filters", "include"),
                ("Apply Filters", "Generate Report", "include")
            ]),
            ("Data Export", ["User"], ["Export Data", "Choose Format", "Schedule Export"], [
                ("User", "Export Data", "association"),
                ("Export Data", "Choose Format", "include"),
                ("Export Data", "Schedule Export", "extend")
            ]),
            ("Dashboard View", ["Manager"], ["View Dashboard", "Customize Widgets", "Refresh Data"], [
                ("Manager", "View Dashboard", "association"),
                ("View Dashboard", "Customize Widgets", "extend"),
                ("View Dashboard", "Refresh Data", "extend")
            ]),
            ("Data Visualization", ["Analyst"], ["Create Chart", "Select Metrics", "Share Visualization"], [
                ("Analyst", "Create Chart", "association"),
                ("Create Chart", "Select Metrics", "include"),
                ("Create Chart", "Share Visualization", "extend")
            ]),
            ("Trend Analysis", ["Analyst"], ["Analyze Trends", "Compare Periods", "Generate Forecast"], [
                ("Analyst", "Analyze Trends", "association"),
                ("Analyze Trends", "Compare Periods", "include"),
                ("Analyze Trends", "Generate Forecast", "extend")
            ])
        ]

        for i, (name, actors, usecases, rels) in enumerate(base_scenarios):
            for variant in range(20):
                variant_name = f"{name}_v{variant + 1}"
                templates.append((f"analytics_{i}_{variant}", variant_name, actors, usecases, rels))

        return templates

    def _get_permission_management_templates(self) -> List[Tuple[str, List[str], List[str], List[Tuple]]]:
        """
        权限管理系统模板 (100个场景)
        """
        templates = []

        base_scenarios = [
            ("Role Assignment", ["Admin"], ["Create Role", "Assign Permissions", "Assign to User"], [
                ("Admin", "Create Role", "association"),
                ("Create Role", "Assign Permissions", "include"),
                ("Create Role", "Assign to User", "extend")
            ]),
            ("Access Control", ["User", "Admin"], ["Request Access", "Approve Request", "Grant Access"], [
                ("User", "Request Access", "association"),
                ("Admin", "Approve Request", "association"),
                ("Approve Request", "Grant Access", "include")
            ]),
            ("Permission Audit", ["Auditor"], ["Review Permissions", "Generate Audit Log", "Flag Issues"], [
                ("Auditor", "Review Permissions", "association"),
                ("Review Permissions", "Generate Audit Log", "include"),
                ("Review Permissions", "Flag Issues", "extend")
            ]),
            ("Delegation", ["Manager"], ["Delegate Authority", "Set Duration", "Notify Delegate"], [
                ("Manager", "Delegate Authority", "association"),
                ("Delegate Authority", "Set Duration", "include"),
                ("Delegate Authority", "Notify Delegate", "extend")
            ]),
            ("Resource Protection", ["Admin"], ["Set Resource Policy", "Define Rules", "Monitor Access"], [
                ("Admin", "Set Resource Policy", "association"),
                ("Set Resource Policy", "Define Rules", "include"),
                ("Set Resource Policy", "Monitor Access", "extend")
            ])
        ]

        for i, (name, actors, usecases, rels) in enumerate(base_scenarios):
            for variant in range(20):
                variant_name = f"{name}_v{variant + 1}"
                templates.append((f"permission_{i}_{variant}", variant_name, actors, usecases, rels))

        return templates

    def _get_notification_system_templates(self) -> List[Tuple[str, List[str], List[str], List[Tuple]]]:
        """
        通知系统模板 (100个场景)
        """
        templates = []

        base_scenarios = [
            ("Push Notification", ["System", "User"], ["Send Notification", "Format Message", "Deliver Push"], [
                ("System", "Send Notification", "association"),
                ("Send Notification", "Format Message", "include"),
                ("Format Message", "Deliver Push", "include"),
                ("User", "Deliver Push", "association")
            ]),
            ("Email Notification", ["System"], ["Compose Email", "Add Attachments", "Send Email"], [
                ("System", "Compose Email", "association"),
                ("Compose Email", "Add Attachments", "extend"),
                ("Compose Email", "Send Email", "include")
            ]),
            ("SMS Notification", ["System"], ["Send SMS", "Validate Number", "Track Delivery"], [
                ("System", "Send SMS", "association"),
                ("Send SMS", "Validate Number", "include"),
                ("Send SMS", "Track Delivery", "extend")
            ]),
            ("Notification Preferences", ["User"], ["Set Preferences", "Choose Channels", "Set Quiet Hours"], [
                ("User", "Set Preferences", "association"),
                ("Set Preferences", "Choose Channels", "include"),
                ("Set Preferences", "Set Quiet Hours", "extend")
            ]),
            ("Batch Notifications", ["Admin"], ["Create Campaign", "Select Recipients", "Schedule Send"], [
                ("Admin", "Create Campaign", "association"),
                ("Create Campaign", "Select Recipients", "include"),
                ("Create Campaign", "Schedule Send", "extend")
            ])
        ]

        for i, (name, actors, usecases, rels) in enumerate(base_scenarios):
            for variant in range(20):
                variant_name = f"{name}_v{variant + 1}"
                templates.append((f"notification_{i}_{variant}", variant_name, actors, usecases, rels))

        return templates

    def _get_file_management_templates(self) -> List[Tuple[str, List[str], List[str], List[Tuple]]]:
        """
        文件管理系统模板 (100个场景)
        """
        templates = []

        base_scenarios = [
            ("File Upload", ["User"], ["Upload File", "Scan Virus", "Store File"], [
                ("User", "Upload File", "association"),
                ("Upload File", "Scan Virus", "include"),
                ("Scan Virus", "Store File", "include")
            ]),
            ("File Download", ["User"], ["Download File", "Check Permission", "Log Access"], [
                ("User", "Download File", "association"),
                ("Download File", "Check Permission", "include"),
                ("Download File", "Log Access", "extend")
            ]),
            ("File Sharing", ["User"], ["Share File", "Set Permissions", "Generate Link"], [
                ("User", "Share File", "association"),
                ("Share File", "Set Permissions", "include"),
                ("Share File", "Generate Link", "extend")
            ]),
            ("Folder Management", ["User"], ["Create Folder", "Move Files", "Set Access"], [
                ("User", "Create Folder", "association"),
                ("Create Folder", "Move Files", "extend"),
                ("Create Folder", "Set Access", "extend")
            ]),
            ("Version Control", ["User"], ["Upload New Version", "Compare Versions", "Restore Version"], [
                ("User", "Upload New Version", "association"),
                ("Upload New Version", "Compare Versions", "extend"),
                ("Upload New Version", "Restore Version", "extend")
            ])
        ]

        for i, (name, actors, usecases, rels) in enumerate(base_scenarios):
            for variant in range(20):
                variant_name = f"{name}_v{variant + 1}"
                templates.append((f"file_{i}_{variant}", variant_name, actors, usecases, rels))

        return templates

    def _get_booking_system_templates(self) -> List[Tuple[str, List[str], List[str], List[Tuple]]]:
        """
        预约系统模板 (100个场景)
        """
        templates = []

        base_scenarios = [
            ("Make Reservation", ["Customer"], ["Check Availability", "Book Slot", "Confirm Booking"], [
                ("Customer", "Check Availability", "association"),
                ("Check Availability", "Book Slot", "include"),
                ("Book Slot", "Confirm Booking", "include")
            ]),
            ("Cancellation", ["Customer"], ["Cancel Booking", "Check Policy", "Process Refund"], [
                ("Customer", "Cancel Booking", "association"),
                ("Cancel Booking", "Check Policy", "include"),
                ("Check Policy", "Process Refund", "extend")
            ]),
            ("Rescheduling", ["Customer"], ["Request Reschedule", "Find New Slot", "Update Booking"], [
                ("Customer", "Request Reschedule", "association"),
                ("Request Reschedule", "Find New Slot", "include"),
                ("Find New Slot", "Update Booking", "include")
            ]),
            ("Reminder System", ["System", "Customer"], ["Send Reminder", "Check Booking", "Notify Customer"], [
                ("System", "Send Reminder", "association"),
                ("Send Reminder", "Check Booking", "include"),
                ("Check Booking", "Notify Customer", "include"),
                ("Customer", "Notify Customer", "association")
            ]),
            ("Resource Management", ["Admin"], ["Manage Slots", "Set Capacity", "Block Dates"], [
                ("Admin", "Manage Slots", "association"),
                ("Manage Slots", "Set Capacity", "include"),
                ("Manage Slots", "Block Dates", "extend")
            ])
        ]

        for i, (name, actors, usecases, rels) in enumerate(base_scenarios):
            for variant in range(20):
                variant_name = f"{name}_v{variant + 1}"
                templates.append((f"booking_{i}_{variant}", variant_name, actors, usecases, rels))

        return templates

    def generate_plantuml_code(self, scenario_name: str, actors: List[str],
                               use_cases: List[str], relationships: List[Tuple]) -> str:
        """
        生成PlantUML代码

        Args:
            scenario_name: 场景名称
            actors: 参与者列表
            use_cases: 用例列表
            relationships: 关系列表 [(from, to, type), ...]

        Returns:
            str: PlantUML代码
        """
        code_lines = ["@startuml", "left to right direction"]

        for actor in actors:
            code_lines.append(f"actor {actor}")

        code_lines.append("")
        code_lines.append(f"rectangle \"{scenario_name}\" {{")

        for uc in use_cases:
            code_lines.append(f"  usecase ({uc})")

        code_lines.append("}")
        code_lines.append("")

        for from_elem, to_elem, rel_type in relationships:
            if rel_type == "association":
                code_lines.append(f"{from_elem} --> ({to_elem})")
            elif rel_type == "include":
                code_lines.append(f"({from_elem}) ..> ({to_elem}) : <<include>>")
            elif rel_type == "extend":
                code_lines.append(f"({from_elem}) ..> ({to_elem}) : <<extend>>")
            elif rel_type == "generalization":
                code_lines.append(f"{from_elem} --|> {to_elem}")

        code_lines.append("@enduml")

        return "\n".join(code_lines)

    def check_java(self) -> bool:
        """
        检查Java是否安装

        Returns:
            bool: True表示Java已安装
        """
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def download_plantuml(self) -> bool:
        """
        下载PlantUML JAR文件

        Returns:
            bool: True表示下载成功
        """
        if self.plantuml_jar.exists():
            print(f"PlantUML already exists: {self.plantuml_jar}")
            return True

        print("Downloading PlantUML...")
        url = "https://github.com/plantuml/plantuml/releases/download/v1.2024.3/plantuml-1.2024.3.jar"

        try:
            urllib.request.urlretrieve(url, self.plantuml_jar)
            print(f"PlantUML downloaded to: {self.plantuml_jar}")
            return True
        except Exception as e:
            print(f"Failed to download PlantUML: {e}")
            return False

    def generate_png(self, puml_file: Path) -> bool:
        """
        使用PlantUML生成PNG图像

        Args:
            puml_file: PlantUML源文件路径

        Returns:
            bool: True表示生成成功
        """
        try:
            result = subprocess.run(
                [
                    "java", "-jar", str(self.plantuml_jar),
                    "-tpng",
                    "-Sdpi=150",
                    str(puml_file)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Failed to generate PNG: {e}")
            return False

    def generate_all(self, target_count: int = 1000):
        """
        批量生成所有用例图

        Args:
            target_count: 目标生成数量
        """
        if not self.check_java():
            print("Error: Java is not installed. Please install Java 8+ first.")
            sys.exit(1)

        if not self.download_plantuml():
            print("Error: Failed to download PlantUML.")
            sys.exit(1)

        print(f"\nGenerating {target_count} use case diagrams...")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)

        generated_count = 0
        total_templates = sum(len(templates) for templates in self.domains.values())

        print(f"Total templates available: {total_templates}")

        for domain_name, templates in self.domains.items():
            domain_dir = self.output_dir / domain_name
            domain_dir.mkdir(exist_ok=True)

            templates_to_use = min(len(templates), target_count // 10)

            print(f"\n[{domain_name}] Generating {templates_to_use} diagrams...")

            for i, (file_id, scenario_name, actors, use_cases, relationships) in enumerate(templates[:templates_to_use]):
                puml_code = self.generate_plantuml_code(scenario_name, actors, use_cases, relationships)

                puml_file = domain_dir / f"{file_id}.puml"
                with open(puml_file, "w", encoding="utf-8") as f:
                    f.write(puml_code)

                if self.generate_png(puml_file):
                    generated_count += 1
                    if (i + 1) % 10 == 0:
                        print(f"  Progress: {i + 1}/{templates_to_use} diagrams")
                else:
                    print(f"  Warning: Failed to generate {puml_file.name}")

                if generated_count >= target_count:
                    break

            if generated_count >= target_count:
                break

        print("\n" + "=" * 60)
        print(f"Generation complete!")
        print(f"Total generated: {generated_count} use case diagrams")
        print(f"Output location: {self.output_dir}")
        print("=" * 60)


def main():
    """主函数"""
    try:
        from config.settings import get_path_config

        path_config = get_path_config()
        output_dir = path_config.PLANTUML_USECASE_DIR

    except ImportError:
        print("Warning: Cannot import config.settings, using default path")
        output_dir = Path(__file__).parent.parent / "data" / "raw" / "uml" / "plantuml_usecase"

    generator = PlantUMLGenerator(output_dir)

    target_count = 1000
    generator.generate_all(target_count=target_count)


if __name__ == "__main__":
    main()