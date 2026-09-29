#!/bin/bash
# 最小可跑通：顺序执行各任务，每个任务带超时保护，失败/超时则跳过继续下一个。
# 全程离线（HF_HUB_OFFLINE=1），使用本地模型目录，不依赖网络。
# 用法: bash _minimal_data/run_all.sh

ROOT="D:/new plm"
PY="D:/Anaconda/envs/plm/python.exe"
BERT="$ROOT/_models/bert-base-chinese"
LOG="$ROOT/_minimal_data/runlogs"
mkdir -p "$LOG"

export HF_HUB_OFFLINE=1
export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

SUMMARY="$LOG/_summary.tsv"
: > "$SUMMARY"
printf "%-28s\t%-10s\t%-8s\t%s\n" "TASK" "STATUS" "SECONDS" "NOTE" >> "$SUMMARY"

run_task() {
    local name="$1"; local limit="$2"; local dir="$3"; shift 3
    local start=$(date +%s)
    echo "=============================================================="
    echo "[START] $name   (超时上限 ${limit}s)"
    echo "=============================================================="
    ( cd "$dir" && timeout "$limit" "$PY" -u "$@" ) > "$LOG/$name.log" 2>&1
    local code=$?
    local dur=$(( $(date +%s) - start ))

    local status note
    if [ $code -eq 0 ]; then
        status="OK"
        note=$(grep -aE "best F1 performence has been updated" "$LOG/$name.log" | tail -1 | tr -d '\r')
        [ -z "$note" ] && note=$(grep -aE "Evaluation" "$LOG/$name.log" | tail -1 | tr -d '\r')
    elif [ $code -eq 124 ]; then
        status="TIMEOUT"; note="超过 ${limit}s 被中断"
    else
        status="FAIL($code)"
        note=$(grep -aiE "error|Traceback" "$LOG/$name.log" | tail -1 | cut -c1-110 | tr -d '\r')
    fi

    printf "%-28s\t%-10s\t%-8s\t%s\n" "$name" "$status" "$dur" "$note" >> "$SUMMARY"
    printf "[%s] %-26s %8ss  %s\n" "$status" "$name" "$dur" "$note"
}

# ---------------------------------------------------------------- 任务队列
# 1. PET (62 条训练样本)
run_task "PET" 2400 "$ROOT/prompt_tasks/PET" pet.py \
    --model "$BERT" \
    --train_path data/comment_classify/train.txt \
    --dev_path data/comment_classify/dev.txt \
    --save_dir checkpoints/comment_classify \
    --img_log_dir logs/comment_classify --img_log_name BERT \
    --verbalizer data/comment_classify/verbalizer.txt \
    --prompt_file data/comment_classify/prompt.txt \
    --batch_size 8 --max_seq_len 128 --valid_steps 40 --logging_steps 5 \
    --num_train_epochs 20 --max_label_len 2 --rdrop_coef 5e-2 --device cpu

# 2. 文本匹配 - PointWise 单塔 (1416 条)
run_task "matching_pointwise" 2400 "$ROOT/text_matching/supervised" train_pointwise.py \
    --model "$BERT" \
    --train_path data/comment_classify/train.txt --dev_path data/comment_classify/dev.txt \
    --save_dir checkpoints/comment_classify/pointwise \
    --img_log_dir logs/comment_classify --img_log_name PointWise \
    --batch_size 8 --max_seq_len 128 --valid_steps 100 --logging_steps 20 \
    --num_train_epochs 2 --device cpu

# 3. 文本匹配 - DSSM 双塔
run_task "matching_dssm" 2400 "$ROOT/text_matching/supervised" train_dssm.py \
    --model "$BERT" \
    --train_path data/comment_classify/train.txt --dev_path data/comment_classify/dev.txt \
    --save_dir checkpoints/comment_classify/dssm \
    --img_log_dir logs/comment_classify --img_log_name DSSM \
    --batch_size 8 --max_seq_len 128 --valid_steps 100 --logging_steps 20 \
    --num_train_epochs 2 --device cpu

# 4. 文本匹配 - Sentence Transformer
run_task "matching_senttrans" 2400 "$ROOT/text_matching/supervised" train_sentence_transformer.py \
    --model "$BERT" \
    --train_path data/comment_classify/train.txt --dev_path data/comment_classify/dev.txt \
    --save_dir checkpoints/comment_classify/sentence_transformer \
    --img_log_dir logs/comment_classify --img_log_name SentenceTransformer \
    --batch_size 8 --max_seq_len 128 --valid_steps 100 --logging_steps 20 \
    --num_train_epochs 2 --device cpu

# 5. UIE 信息抽取 (2087 条，权重已本地备好)
run_task "UIE" 3600 "$ROOT/UIE" train.py \
    --pretrained_model uie-base-zh \
    --save_dir checkpoints/DuIE \
    --train_path data/DuIE/train.txt --dev_path data/DuIE/dev.txt \
    --img_log_dir logs --img_log_name "UIE Base" \
    --batch_size 16 --max_seq_len 128 --learning_rate 5e-5 \
    --num_train_epochs 2 --logging_steps 20 --valid_steps 50 --device cpu

# 6. RLHF 奖励模型 (采样 3000 条)
run_task "RLHF_reward" 3600 "$ROOT/RLHF" train_reward_model.py \
    --model "$BERT" \
    --train_path "$ROOT/_minimal_data/rlhf_reward_train.tsv" \
    --dev_path "$ROOT/_minimal_data/rlhf_reward_dev.tsv" \
    --save_dir checkpoints/reward_model/sentiment_analysis \
    --img_log_dir logs/reward_model/sentiment_analysis --img_log_name "Reward Model" \
    --batch_size 16 --max_seq_len 128 --learning_rate 1e-5 \
    --valid_steps 50 --logging_steps 10 --num_train_epochs 1 --device cpu

# 7. SimCSE 无监督 (采样 2 万条)
run_task "SimCSE" 5400 "$ROOT/text_matching/unsupervised/simcse" train.py \
    --model "$BERT" \
    --train_path "$ROOT/_minimal_data/simcse_train.txt" \
    --dev_path "$ROOT/_minimal_data/simcse_dev.tsv" \
    --save_dir checkpoints/LCQMC \
    --img_log_dir logs/LCQMC --img_log_name ESimCSE \
    --learning_rate 1e-5 --dropout 0.3 \
    --batch_size 16 --max_seq_len 64 --valid_steps 400 --logging_steps 100 \
    --num_train_epochs 1 --device cpu

echo
echo "=============================================================="
echo "全部任务执行完毕，汇总："
echo "=============================================================="
cat "$SUMMARY"
