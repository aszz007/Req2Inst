#专家选择
"""
专家路由模块
功能：根据意图和内容特征选择合适的专家
"""


class ExpertRouter:
    """专家路由器"""

    def __init__(self):
        # 定义可用的专家
        self.experts = {
            'Image_Annotation_Expert': {
                'name': 'Image_Annotation_Expert',
                'description': '图像标注任务专家',
                'specialization': '处理图像中的目标标注、区域标注等任务'
            },
            'Text_Extraction_Expert': {
                'name': 'Text_Extraction_Expert',
                'description': '文本提取任务专家',
                'specialization': '处理文本中的实体提取、信息抽取等任务'
            },
            'Recognition_Expert': {
                'name': 'Recognition_Expert',
                'description': '识别任务专家',
                'specialization': '处理图像或文本的分类、识别等任务'
            },
            'Generation_Expert': {
                'name': 'Generation_Expert',
                'description': '生成任务专家',
                'specialization': '处理内容生成、创作等任务'
            },
            'Complex_Task_Expert': {
                'name': 'Complex_Task_Expert',
                'description': '复杂任务专家',
                'specialization': '处理高复杂度、多步骤的综合任务'
            },
            'General_Expert': {
                'name': 'General_Expert',
                'description': '通用任务专家',
                'specialization': '处理常规任务和未分类任务'
            }
        }

    def select_expert(self, intent, content_features):
        """
        选择合适的专家

        Args:
            intent: 意图类别
            content_features: 内容特征字典

        Returns:
            dict: 选中的专家信息
        """
        modality = content_features.get('modality', 'none')
        complexity = content_features.get('complexity', 'medium')

        # 路由逻辑
        expert_name = self._route_logic(intent, modality, complexity)
        expert_info = self.experts[expert_name].copy()

        # 添加选择原因
        expert_info['reason'] = self._generate_reason(intent, modality, complexity)

        return expert_info

    def _route_logic(self, intent, modality, complexity):
        """
        核心路由逻辑

        基于文档要求的路由规则
        """
        # 规则1：标注 + 图像 -> Image_Annotation_Expert
        if intent == '标注' and modality == 'image':
            return 'Image_Annotation_Expert'

        # 规则2：提取 + 文本 -> Text_Extraction_Expert
        elif intent == '提取' and modality == 'text':
            return 'Text_Extraction_Expert'

        # 规则3：识别任务 -> Recognition_Expert
        elif intent == '识别':
            return 'Recognition_Expert'

        # 规则4：生成/创作 -> Generation_Expert
        elif intent in ['生成', '创作']:
            return 'Generation_Expert'

        # 规则5：高复杂度 -> Complex_Task_Expert
        elif complexity == 'high':
            return 'Complex_Task_Expert'

        # 规则6：其他情况 -> General_Expert
        else:
            return 'General_Expert'

    def _generate_reason(self, intent, modality, complexity):
        """生成选择原因说明"""
        reasons = []

        if intent != '通用':
            reasons.append(f"意图为'{intent}'")

        if modality != 'none':
            reasons.append(f"内容类型为'{modality}'")

        if complexity == 'high':
            reasons.append(f"复杂度为'{complexity}'")

        if not reasons:
            return "默认选择"

        return "，".join(reasons)

    def get_all_experts(self):
        """获取所有可用专家"""
        return list(self.experts.keys())


# 测试代码
if __name__ == "__main__":
    router = ExpertRouter()

    test_cases = [
        {
            'intent': '标注',
            'features': {'modality': 'image', 'complexity': 'medium'}
        },
        {
            'intent': '提取',
            'features': {'modality': 'text', 'complexity': 'low'}
        },
        {
            'intent': '识别',
            'features': {'modality': 'image', 'complexity': 'low'}
        },
        {
            'intent': '生成',
            'features': {'modality': 'none', 'complexity': 'medium'}
        }
    ]

    print("专家选择测试:")
    print("-" * 40)
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"  意图: {test['intent']}")
        print(f"  特征: {test['features']}")

        result = router.select_expert(test['intent'], test['features'])
        print(f"  选择: {result['name']}")
        print(f"  原因: {result['reason']}")