#!/bin/bash
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"
PYTHON="/root/miniconda3/envs/instruction_generator/bin/python"

# Python过滤器：将\r回车符还原为终端实际展示效果（仅保留每行最终状态）
# 全程使用二进制模式读写，避免管道环境下stdout编码不是UTF-8导致中文乱码
_CR_FILTER='
import sys
out = sys.stdout.buffer
cur = b""
while True:
    ch = sys.stdin.buffer.read(1)
    if not ch:
        if cur:
            out.write(cur + b"\n")
            out.flush()
        break
    if ch == b"\r":
        cur = b""
    elif ch == b"\n":
        out.write(cur + b"\n")
        out.flush()
        cur = b""
    else:
        cur += ch
'

run_task() {
    local name=$1
    local script=$2
    echo "[START] $name - $(date)" | tee -a "$LOG_FILE"

    local _ec_file
    _ec_file=$(mktemp)

    # tee 将原始字节流同时送往：
    #   1. 终端 stdout（\r 由终端自然处理，进度条实时刷新）
    #   2. 进程替换：过滤器折叠 \r，只追加干净结果到日志文件
    { $PYTHON $script; echo $? > "$_ec_file"; } 2>&1 | \
        tee >(python3 -u -c "$_CR_FILTER" >> "$LOG_FILE")

    local exit_code
    exit_code=$(cat "$_ec_file" 2>/dev/null || echo 1)
    rm -f "$_ec_file"

    if [ "$exit_code" -eq 0 ]; then
        echo "[DONE]  $name - $(date)" | tee -a "$LOG_FILE"
    else
        echo "[FAIL]  $name (exit code: $exit_code) - $(date)" | tee -a "$LOG_FILE"
    fi
}

# ---- 训练任务列表 ----
#run_task "Lora MoE Text Expert"            "scripts/training/lora_moe/train_text_expert.py"
#run_task "Lora MoE Image Expert"           "scripts/training/lora_moe/train_image_expert.py"
#run_task "Lora MoE UML Expert"             "scripts/training/lora_moe/train_uml_expert.py"
#run_task "Lora MoE General Expert"         "scripts/training/lora_moe/train_general_expert.py"
#run_task "Lora Single"                     "scripts/training/lora_single/train_unified_expert.py"
#run_task "Full Finetuning Text Expert"     "scripts/training/full_finetuning/train_text_expert.py"
#run_task "Full Finetuning Image Expert"    "scripts/training/full_finetuning/train_image_expert.py"
#run_task "Full Finetuning UML Expert"      "scripts/training/full_finetuning/train_uml_expert.py"
#run_task "Full Finetuning General Expert"  "scripts/training/full_finetuning/train_general_expert.py"
#run_task "Experiment 1"                    "scripts/evaluation/experiments/exp1_baseline_comparison.py"
#run_task "Experiment 2"                    "scripts/evaluation/experiments/exp2_compare_finetuning_methods.py"
#run_task "Experiment 3"                    "scripts/evaluation/experiments/exp3_moe_architecture_validation.py"
#run_task "Experiment 4"                    "scripts/evaluation/experiments/exp4_lora_hyperparameter_optimization.py"
#run_task "Experiment 5"                    "scripts/evaluation/experiments/exp5_data_efficiency_analysis.py"
#run_task "Experiment 6"                    "scripts/evaluation/experiments/exp6_fewshot_vs_finetuning.py"
#run_task "Experiment 7"                    "scripts/evaluation/experiments/exp7_uml_hyperparameter_optimization.py"
#run_task "Experiment 8"                    "scripts/evaluation/experiments/exp8_inference_efficiency.py"
#run_task "Experiment 9"                    "scripts/evaluation/experiments/exp9_routing_strategy.py --all"
#run_task "Experiment 10"                   "scripts/evaluation/experiments/exp10_advanced_routing.py --all"
#run_task "Experiment 11"                   "scripts/evaluation/experiments/exp11_ablation_optimization.py --all"