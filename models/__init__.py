"""
模型模块
封装基础模型和LoRA-MoE系统
版本: 2.0（支持LoRA动态加载）
"""

# 导入语言模型
from .language_model import (
    LanguageModel,
    InstructionGenerator,
)

# 导入视觉模型
from .vision_model import (
    VisionModel,
)

__all__ = [
    # 语言模型
    'LanguageModel',
    'InstructionGenerator',

    # 视觉模型
    'VisionModel',
]

# 版本信息
__version__ = '2.0.0'
__author__ = 'Crowdsourcing Instruction Generator Team'

# 快速使用示例（注释）
"""
使用示例：

1. 语言模型基础使用：
    from models import LanguageModel
    
    model = LanguageModel(use_4bit=True)
    response = model.generate("你的prompt")

2. 语言模型 + LoRA：
    from models import InstructionGenerator
    from config import get_path_config
    
    generator = InstructionGenerator(use_4bit=True)
    
    # 方式1: 使用专家名称加载
    generator.load_expert('text_expert')
    
    # 方式2: 使用完整路径加载
    path_cfg = get_path_config()
    lora_path = path_cfg.get_expert_weight_path('text_expert')
    generator.language_model.load_lora_from_path(str(lora_path))
    
    # 生成指令
    instruction = generator.generate(prompt)
    
    # 卸载LoRA
    generator.unload_expert()

3. 视觉模型预处理：
    from models import VisionModel
    
    model = VisionModel()
    
    # 图像识别（预处理）
    result = model.recognize_image('image.jpg')
    
    # UML识别（预处理）
    result = model.recognize_uml('uml.png')

4. 视觉模型 + LoRA推理：
    from models import VisionModel
    
    model = VisionModel()
    
    # 加载LoRA
    model.load_lora_from_path('lora_weights/experts/image_expert_qwen2.5')
    
    # 通用生成
    instruction = model.generate(
        prompt="根据图像生成标注指令",
        image_path='image.jpg'
    )
    
    # 卸载LoRA
    model.unload_lora()

5. 查看模型状态：
    from models import InstructionGenerator, VisionModel
    
    generator = InstructionGenerator()
    status = generator.get_expert_status()
    print(f"LoRA已加载: {status['is_loaded']}")
    print(f"当前路径: {status['current_path']}")
    
    vision_model = VisionModel()
    status = vision_model.get_lora_status()
    print(f"LoRA已加载: {status['is_loaded']}")
"""