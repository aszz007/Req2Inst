#!/usr/bin/env python3
"""
Quick diagnostic script for p_tuning / prompt_tuning inference precision issue.

Runs 5 samples with 4bit and FP16 configurations, then scores each config
to determine whether the checkpoint produces valid output in each mode.

Usage:
    python scripts/evaluation/diagnose_soft_prompt_inference.py \
        --method p_tuning \
        --expert-type general

    python scripts/evaluation/diagnose_soft_prompt_inference.py \
        --method prompt_tuning \
        --expert-type text
"""

import sys
import argparse
import re
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_path_config
from src.utils.logger import get_logger

logger = get_logger('diagnose.soft_prompt')

N_SAMPLES = 5

METHOD_CKPT_MAP = {
    'p_tuning': lambda cfg, et: str(cfg.PTUNING_CKPTS[et]),
    'prompt_tuning': lambda cfg, et: str(cfg.PROMPT_TUNING_CKPTS[et]),
}

EXPERT_CLASS_MAP = {
    'text': 'TextExpert',
    'image': 'ImageExpert',
    'uml': 'UMLExpert',
    'general': 'GeneralExpert',
}


def _load_expert(expert_type, ckpt_path, use_4bit):
    from src.experts import TextExpert, ImageExpert, UMLExpert, GeneralExpert
    cls = {'text': TextExpert, 'image': ImageExpert,
           'uml': UMLExpert, 'general': GeneralExpert}[expert_type]
    expert = cls(lora_path=ckpt_path, use_4bit=use_4bit)
    ok = expert.load_model()
    return expert if ok else None


def _load_samples(expert_type, n):
    from src.training.data_loader import (
        TextDatasetLoader, ImageDatasetLoader, UMLDatasetLoader,
        GeneralDatasetLoader, split_dataset_for_expert,
    )
    if expert_type == 'text':
        data = TextDatasetLoader().load_csv_files()
    elif expert_type == 'image':
        data = ImageDatasetLoader().load_csv_file()
    elif expert_type == 'uml':
        data = UMLDatasetLoader().load_csv_file()
    else:
        data = GeneralDatasetLoader().load_all_data()
    _, _, test_data = split_dataset_for_expert(data, expert_type)
    return test_data[:n]


def _score_output(text):
    """
    Score a generated text for validity.
    Returns a tuple (score 0-4, reasons list).

    Scoring criteria (1 point each):
      1. Non-empty output
      2. Contains at least one English word (>= 3 chars)
      3. Contains at least one of the three expected section headers
      4. No multi-script garbage (Korean/Arabic/CJK mixed in unexpected positions)
    """
    if not text or not text.strip():
        return 0, ['empty output']

    score = 1
    reasons = ['non-empty']

    english_words = re.findall(r'[a-zA-Z]{3,}', text)
    if len(english_words) >= 3:
        score += 1
        reasons.append(f'{len(english_words)} English words')
    else:
        reasons.append(f'only {len(english_words)} English words (expected >= 3)')

    headers = ['Definition:', 'Emphasis', 'Things to Avoid:']
    found = [h for h in headers if h in text]
    if found:
        score += 1
        reasons.append(f'headers found: {found}')
    else:
        reasons.append('no section headers found')

    garbage_chars = re.findall(r'[\u1100-\u11ff\u3040-\u30ff\u3400-\u4dbf'
                               r'\u4e00-\u9fff\uac00-\ud7af\u0600-\u06ff]', text)
    if len(garbage_chars) == 0:
        score += 1
        reasons.append('no garbage multi-script chars')
    else:
        reasons.append(f'{len(garbage_chars)} garbage chars: {"".join(garbage_chars[:8])}')

    return score, reasons


def _run_config(expert_type, ckpt_path, samples, use_4bit):
    label = '4bit' if use_4bit else 'FP16'
    logger.info(f'  Loading with {label}...')
    expert = _load_expert(expert_type, ckpt_path, use_4bit)
    if expert is None:
        logger.error(f'  Model load failed for {label}')
        return None

    inputs = [s['input'] for s in samples]
    try:
        preds = expert.batch_generate_instruction(inputs, batch_size=1)
    except Exception as e:
        logger.error(f'  Inference failed ({label}): {e}')
        expert.unload_model()
        return None
    finally:
        expert.unload_model()

    results = []
    total_score = 0
    for i, (pred, sample) in enumerate(zip(preds, samples)):
        score, reasons = _score_output(pred)
        total_score += score
        results.append({
            'sample': i + 1,
            'score': score,
            'reasons': reasons,
            'preview': (pred[:120] + '...') if len(pred) > 120 else pred,
        })
        logger.info(f'  Sample {i+1}: score={score}/4 | {" | ".join(reasons)}')
        logger.info(f'  Preview: {results[-1]["preview"]}')

    avg = total_score / len(samples) if samples else 0
    logger.info(f'  [{label}] Average score: {avg:.1f}/4')
    return {'config': label, 'avg_score': avg, 'results': results}


def main():
    parser = argparse.ArgumentParser(description='Diagnose p_tuning/prompt_tuning inference precision')
    parser.add_argument('--method', required=True, choices=['p_tuning', 'prompt_tuning'])
    parser.add_argument('--expert-type', required=True, choices=['text', 'image', 'uml', 'general'])
    parser.add_argument('--n-samples', type=int, default=N_SAMPLES,
                        help='Number of test samples (default: 5)')
    parser.add_argument('--skip-4bit', action='store_true',
                        help='Skip 4bit test (saves time if you already know it fails)')
    args = parser.parse_args()

    cfg = get_path_config()
    ckpt_path = METHOD_CKPT_MAP[args.method](cfg, args.expert_type)
    logger.info(f'Checkpoint: {ckpt_path}')
    logger.info(f'Method: {args.method} | Expert: {args.expert_type} | Samples: {args.n_samples}')

    logger.info('Loading test samples...')
    samples = _load_samples(args.expert_type, args.n_samples)
    logger.info(f'Loaded {len(samples)} samples')

    configs_to_test = []
    if not args.skip_4bit:
        configs_to_test.append(True)
    configs_to_test.append(False)

    report = {}
    for use_4bit in configs_to_test:
        label = '4bit' if use_4bit else 'FP16'
        logger.info(f'\n{"="*60}')
        logger.info(f'Testing config: {label}')
        logger.info('='*60)
        result = _run_config(args.expert_type, ckpt_path, samples, use_4bit)
        if result:
            report[label] = result

    logger.info(f'\n{"="*60}')
    logger.info('DIAGNOSIS SUMMARY')
    logger.info('='*60)
    for label, r in report.items():
        verdict = 'PASS' if r['avg_score'] >= 2.5 else 'FAIL'
        logger.info(f'  {label}: avg_score={r["avg_score"]:.1f}/4  [{verdict}]')

    fp16_ok = report.get('FP16', {}).get('avg_score', 0) >= 2.5
    bit4_ok = report.get('4bit', {}).get('avg_score', 0) >= 2.5

    logger.info('')
    if not bit4_ok and fp16_ok:
        logger.info('CONCLUSION: Soft prompt + 4bit quantization incompatible.')
        logger.info('ACTION: Use FP16 inference for this method.')
        logger.info('        exp2_compare_finetuning_methods.py already contains this fix via')
        logger.info('        METHODS_REQUIRE_FP16 constant. Re-run exp2 with --force-regenerate.')
    elif bit4_ok and fp16_ok:
        logger.info('CONCLUSION: Both configs produce valid output.')
        logger.info('ACTION: No change needed. Checkpoint is healthy.')
    elif not bit4_ok and not fp16_ok:
        logger.info('CONCLUSION: Both configs fail. Checkpoint may be corrupt or undertrained.')
        logger.info('ACTION: Check training logs for this method/expert. Consider retraining.')
    else:
        logger.info('CONCLUSION: 4bit works but FP16 does not (unexpected).')
        logger.info('ACTION: Keep using 4bit. Investigate FP16 memory or dtype issues.')


if __name__ == '__main__':
    main()