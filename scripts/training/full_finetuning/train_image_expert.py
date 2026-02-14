"""
Full Fine-tuning Image Expert训练脚本

功能：使用高rank LoRA模拟全参数微调
环境：instruction_generator（transformers==4.51.0）
基础模型：Qwen3-8B
方法：High-rank LoRA (rank=64) 模拟全参数微调
输出：checkpoints/full_finetuning/image_expert/

对比实验说明：
  - Full Fine-tuning vs LoRA：验证参数效率和性能权衡
  - 使用rank=64的LoRA覆盖更多层模拟全参数微调
  - 相比真正的全参数微调更稳定且显存友好

使用方法：
  python scripts/training/full_finetuning/train_image_expert.py

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
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from config.settings import (
    get_path_config,
    get_training_config,
    get_full_finetuning_config
)
from src.training.data_loader import (
    ImageDatasetLoader,
    InstructionDataset,
    InstructionDataCollator,
    split_dataset_for_expert
)
from models.prompt_templates.image_template import ImageInstructionTemplate
from src.utils.logger import get_logger

logger = get_logger('training.full_finetuning.image_expert')


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
    print(" " * 12 + "准全参数微调 Image Expert训练 (High-rank LoRA)")
    print("=" * 80)
    print()


def main():
    """主训练流程"""
    parser = argparse.ArgumentParser(description='使用高rank LoRA模拟全参数微调')
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
    full_ft_cfg = get_full_finetuning_config()

    # 检测RTX 4090
    is_rtx4090 = detect_rtx4090()
    if is_rtx4090:
        logger.info("检测到RTX 4090，启用优化配置")

    # 打印实验说明
    print("=" * 80)
    print("对比实验：准全参数微调 vs LoRA")
    print("=" * 80)
    print("方法：High-rank LoRA（模拟全参数微调）")
    print("配置：")
    print(f"  - LoRA Rank: {full_ft_cfg.lora_rank} (vs 标准LoRA的8)")
    print(f"  - LoRA Alpha: {full_ft_cfg.lora_alpha}")
    print(f"  - Target Modules: {len(full_ft_cfg.target_modules)}层")
    print(f"  - Learning Rate: {full_ft_cfg.learning_rate} (更保守)")
    print("说明：使用大rank LoRA覆盖更多层，接近全参数微调效果")
    print("=" * 80)
    print()

    # 1. 加载数据
    logger.info("加载Image数据集...")
    data_loader = ImageDatasetLoader()
    raw_data = data_loader.load_csv_file()

    # 划分数据集
    train_data, val_data, _ = split_dataset_for_expert(raw_data, 'image')
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

    # 3. 应用高rank LoRA配置（模拟全参数微调）
    logger.info("配置高rank LoRA...")
    peft_config = LoraConfig(
        r=full_ft_cfg.lora_rank,  # rank=64
        lora_alpha=full_ft_cfg.lora_alpha,  # alpha=128
        target_modules=full_ft_cfg.target_modules,  # 覆盖attention + FFN层
        lora_dropout=full_ft_cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. 准备数据集
    logger.info("准备训练数据集...")
    template = ImageInstructionTemplate()

    train_dataset = InstructionDataset(
        data=train_data,
        tokenizer=tokenizer,
        template=template,
        max_length=2048
    )
    val_dataset = InstructionDataset(
        data=val_data,
        tokenizer=tokenizer,
        template=template,
        max_length=2048
    )

    data_collator = InstructionDataCollator(tokenizer=tokenizer)

    # 5. 设置训练参数（更保守的配置）
    output_dir = path_cfg.FULL_FINETUNING_CKPTS['image']
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=full_ft_cfg.num_epochs,  # 2 epochs（防止过拟合）
        per_device_train_batch_size=full_ft_cfg.batch_size,  # 4（显存限制）
        per_device_eval_batch_size=full_ft_cfg.batch_size,
        gradient_accumulation_steps=full_ft_cfg.gradient_accumulation_steps,  # 4
        learning_rate=full_ft_cfg.learning_rate,  # 1e-4（比LoRA更小）
        weight_decay=full_ft_cfg.weight_decay,
        warmup_ratio=full_ft_cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=train_cfg.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=is_rtx4090,
        fp16=not is_rtx4090,
        max_grad_norm=full_ft_cfg.max_grad_norm,  # 0.5（更严格的梯度裁剪）
        dataloader_num_workers=4 if is_rtx4090 else 2,
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
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    # 7. 保存最终模型
    logger.info(f"保存模型至: {output_dir}")
    trainer.save_model(output_dir)

    print()
    print("=" * 80)
    print(" " * 25 + "训练成功完成！")
    print("=" * 80)
    print(f"准全参数微调权重已保存至: {output_dir}")
    print()
    print("性能对比：")
    print("  - 可训练参数：约为标准LoRA的8倍")
    print("  - 预期效果：接近真正的全参数微调")
    print("  - 优势：更稳定、显存友好、训练更快")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())