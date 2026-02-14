"""
P-Tuning v2 Text Expert训练脚本

功能：使用P-Tuning v2方法训练Text Expert
环境：instruction_generator（transformers==4.51.0）
基础模型：Qwen3-8B
方法：P-Tuning v2（前缀微调）
输出：checkpoints/p_tuning/text_expert/

对比实验说明：
  - P-Tuning v2 vs LoRA：验证不同参数高效微调方法的效果
  - 使用Prefix tokens（20个virtual tokens）
  - 通过MLP编码器学习任务特定的前缀表示

使用方法：
  python scripts/training/p_tuning/train_text_expert.py

作者：Comparative Training System
日期：2025-02-15
"""

import sys
import argparse
import torch
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    PrefixTuningConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from config.settings import (
    get_path_config,
    get_training_config,
    get_ptuning_config
)
from src.training.data_loader import (
    TextDatasetLoader,
    InstructionDataset,
    InstructionDataCollator,
    split_dataset_for_expert
)
from models.prompt_templates.text_template import TextInstructionTemplate
from src.utils.logger import get_logger

logger = get_logger('training.p_tuning.text_expert')


def detect_rtx4090() -> bool:
    """检测是否为RTX 4090显卡"""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return 'RTX 4090' in gpu_name or 'RTX 4090D' in gpu_name
    except:
        pass
    return False


def print_header():
    """打印训练开始的标题"""
    print("=" * 80)
    print(" " * 15 + "P-Tuning v2 Text Expert训练 (Prefix Tuning)")
    print("=" * 80)
    print()


def main():
    """主训练流程"""
    parser = argparse.ArgumentParser(description='使用P-Tuning v2训练Text Expert')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认：True）')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='不使用4bit量化')
    args = parser.parse_args()

    # 打印标题
    print_header()

    # 获取配置
    path_cfg = get_path_config()
    train_cfg = get_training_config()
    ptuning_cfg = get_ptuning_config()

    # 检测RTX 4090
    is_rtx4090 = detect_rtx4090()
    if is_rtx4090:
        logger.info("检测到RTX 4090，启用优化配置")

    # 打印实验说明
    print("=" * 80)
    print("对比实验：P-Tuning v2 vs LoRA")
    print("=" * 80)
    print("方法：P-Tuning v2（Prefix Tuning）")
    print("配置：")
    print(f"  - Virtual Tokens: {ptuning_cfg.num_virtual_tokens}")
    print(f"  - Encoder Hidden Size: {ptuning_cfg.encoder_hidden_size}")
    print(f"  - Encoder Layers: {ptuning_cfg.encoder_num_layers}")
    print("=" * 80)
    print()

    # 1. 加载数据
    logger.info("加载Text数据集...")
    data_loader = TextDatasetLoader()
    raw_data = data_loader.load()

    # 划分数据集
    train_data, val_data, _ = split_dataset_for_expert('text', raw_data)
    logger.info(f"训练样本: {len(train_data)}, 验证样本: {len(val_data)}")

    # 2. 加载模型和分词器
    logger.info("加载基础模型...")
    model_path = path_cfg.get_text_model_path()

    # 4bit量化配置
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if is_rtx4090 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if is_rtx4090 else torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 应用P-Tuning v2配置
    logger.info("配置P-Tuning v2...")
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=ptuning_cfg.num_virtual_tokens,
        encoder_hidden_size=ptuning_cfg.encoder_hidden_size,
        prefix_projection=ptuning_cfg.prefix_projection
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. 准备数据集
    logger.info("准备训练数据集...")
    template = TextInstructionTemplate()

    train_dataset = InstructionDataset(
        data=train_data,
        tokenizer=tokenizer,
        template=template,
        max_length=2048  # 支持长文本
    )
    val_dataset = InstructionDataset(
        data=val_data,
        tokenizer=tokenizer,
        template=template,
        max_length=2048
    )

    data_collator = InstructionDataCollator(tokenizer=tokenizer)

    # 5. 设置训练参数
    output_dir = path_cfg.PTUNING_CKPTS['text']
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_epochs,
        per_device_train_batch_size=8 if is_rtx4090 else train_cfg.batch_size,
        per_device_eval_batch_size=8 if is_rtx4090 else train_cfg.batch_size,
        gradient_accumulation_steps=2 if is_rtx4090 else train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        logging_steps=train_cfg.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=is_rtx4090,
        fp16=not is_rtx4090 and train_cfg.fp16,
        dataloader_num_workers=8 if is_rtx4090 else 2,
        remove_unused_columns=False,
        report_to="none"
    )

    # 6. 创建训练器并开始训练
    logger.info("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()

    # 7. 保存最终模型
    logger.info(f"保存模型至: {output_dir}")
    trainer.save_model(output_dir)

    print()
    print("=" * 80)
    print(" " * 25 + "训练成功完成！")
    print("=" * 80)
    print(f"P-Tuning v2权重已保存至: {output_dir}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())