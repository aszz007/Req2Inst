#内容分析
"""
内容分析模块
功能：分析输入内容的特征
"""


class ContentAnalyzer:
    """内容分析器"""

    def __init__(self):
        pass

    def analyze(self, content, content_type):
        """
        分析内容特征

        Args:
            content: 内容数据（图像描述或文本）
            content_type: 内容类型 ('image', 'text', None)

        Returns:
            dict: 内容特征字典
        """
        features = {
            'modality': self._detect_modality(content, content_type),
            'length': self._calculate_length(content),
            'complexity': self._assess_complexity(content, content_type),
            'has_structure': self._detect_structure(content)
        }

        return features

    def _detect_modality(self, content, content_type):
        """检测内容模态"""
        if content_type:
            return content_type
        elif content is None:
            return 'none'
        else:
            # 简单判断：如果内容很长，可能是文本
            if isinstance(content, str) and len(content) > 50:
                return 'text'
            else:
                return 'mixed'

    def _calculate_length(self, content):
        """计算内容长度"""
        if content is None:
            return 0

        if isinstance(content, str):
            return len(content)
        else:
            return 0

    def _assess_complexity(self, content, content_type):
        """评估内容复杂度"""
        if content is None:
            return 'low'

        # 基于长度的简单评估
        length = self._calculate_length(content)

        if content_type == 'image':
            # 图像：通过描述长度判断
            if length > 100:
                return 'high'
            elif length > 50:
                return 'medium'
            else:
                return 'low'

        elif content_type == 'text':
            # 文本：通过字符数判断
            if length > 500:
                return 'high'
            elif length > 200:
                return 'medium'
            else:
                return 'low'

        return 'medium'

    def _detect_structure(self, content):
        """检测是否有结构化内容"""
        if content is None:
            return False

        if not isinstance(content, str):
            return False

        # 检测常见结构化标志
        structure_markers = ['表格', '清单', '列表', '编号', '1.', '2.', '•', '-']

        for marker in structure_markers:
            if marker in content:
                return True

        return False


# 测试代码
if __name__ == "__main__":
    analyzer = ContentAnalyzer()

    test_cases = [
        {
            "content": "一张街景图片，包含3辆汽车和2个交通标志",
            "type": "image"
        },
        {
            "content": "这是一篇很长的文章" * 30,
            "type": "text"
        },
        {
            "content": None,
            "type": None
        }
    ]

    print("内容分析测试:")
    print("-" * 40)
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        features = analyzer.analyze(test['content'], test['type'])
        for key, value in features.items():
            print(f"  {key}: {value}")