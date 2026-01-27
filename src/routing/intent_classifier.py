#意图识别
"""
意图识别模块
功能：识别用户问题的核心意图
"""


class IntentClassifier:
    """意图分类器"""

    def __init__(self):
        # 定义意图关键词映射
        self.intent_keywords = {
            '标注': ['标注', '标出', '圈出', '框出', '画出', '标记'],
            '提取': ['提取', '找出', '抽取', '提炼', '摘取'],
            '识别': ['识别', '判断', '分类', '检测', '辨别'],
            '生成': ['生成', '创作', '撰写', '编写', '写'],
            '修正': ['纠错', '修正', '检查', '校对', '改错', '纠正'],
            '比较': ['比较', '对比', '区分', '差异'],
            '描述': ['描述', '说明', '解释', '介绍']
        }

    def classify(self, question):
        """
        分类用户问题的意图

        Args:
            question: 用户问题字符串

        Returns:
            str: 意图类别
        """
        question_lower = question.lower()

        # 遍历所有意图，检查关键词
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return intent

        # 如果没有匹配到，返回通用
        return '通用'

    def get_all_intents(self):
        """获取所有支持的意图类别"""
        return list(self.intent_keywords.keys()) + ['通用']


# 测试代码
if __name__ == "__main__":
    classifier = IntentClassifier()

    test_questions = [
        "标出图中的所有汽车",
        "提取文档中的关键信息",
        "识别这是什么物体",
        "生成一段描述",
        "帮我纠错这段文字"
    ]

    print("意图识别测试:")
    print("-" * 40)
    for q in test_questions:
        intent = classifier.classify(q)
        print(f"问题: {q}")
        print(f"意图: {intent}\n")