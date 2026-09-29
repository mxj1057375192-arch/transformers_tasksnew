#!/bin/bash
# 收尾队列：重跑前面失败/超时的任务，让它们干净地跑完（exit 0）。
#   - matching_dssm / matching_senttrans：上次撞 2400s 超时，改成 1 个 epoch
#   - SimCSE：上次设了离线，evaluate 库找不到指标脚本；改为联网
# 用法: bash _minimal_data/run_cleanup.sh

ROOT="D:/new plm"
PY="D:/Anaconda/envs/plm/python.exe"
BERT="$ROOT/_models/bert-base-chinese"
LOG="$ROOT/_minimal_data/runlogs"
mkdir -p "$LOG"

export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

SUMMARY="$LOG/_summary4.tsv"
: > "$SUMMARY"
printf "%-28s\t%-10s\t%-8s\t%s\n" "TASK" "STATUS" "SECONDS" "NOTE" >> "$SUMMARY"

run_task() {
    local name="$1"; local limit="$2"; local dir="$3"; shift 3
    local start=$(date +%s)
    echo "=============================================================="
    echo "[START] $name   (超时上限 ${limit}s)"
    echo "=============================================================="
    ( cd "$dir" && HF_ENDPOINT=https://aifasthub.com timeout "$limit" "$PY" -u "$@" ) > "$LOG/$name.log" 2>&1
    local code=$?
    local dur=$(( $(date +%s) - start ))
    local status note
    if [ $code -eq 0 ]; then
        status="OK"
        note=$(grep -aE "best F1 performence has been updated|Evaluation" "$LOG/$name.log" | tail -1 | tr -d '\r')
    elif [ $code -eq 124 ]; then
        status="TIMEOUT"; note="超过 ${limit}s"
    else
        status="FAIL($code)"
        note=$(grep -aiE "error|Traceback" "$LOG/$name.log" | tail -1 | cut -c1-110 | tr -d '\r')
    fi
    printf "%-28s\t%-10s\t%-8s\t%s\n" "$name" "$status" "$dur" "$note" >> "$SUMMARY"
    printf "[%s] %-26s %8ss  %s\n" "$status" "$name" "$dur" "$note"
}

run_task "matching_dssm" 2400 "$ROOT/text_matching/supervised" train_dssm.py \
    --model "$BERT" \
    --train_path data/comment_classify/train.txt --dev_path data/comment_classify/dev.txt \
    --save_dir checkpoints/comment_classify/dssm \
    --img_log_dir logs/comment_classify --img_log_name DSSM \
    --batch_size 8 --max_seq_len 128 --valid_steps 100 --logging_steps 20 \
    --num_train_epochs 1 --device cpu

run_task "matching_senttrans" 2400 "$ROOT/text_matching/supervised" train_sentence_transformer.py \
    --model "$BERT" \
    --train_path data/comment_classify/train.txt --dev_path data/comment_classify/dev.txt \
    --save_dir checkpoints/comment_classify/sentence_transformer \
    --img_log_dir logs/comment_classify --img_log_name SentenceTransformer \
    --batch_size 8 --max_seq_len 128 --valid_steps 100 --logging_steps 20 \
    --num_train_epochs 1 --device cpu

run_task "SimCSE" 5400 "$ROOT/text_matching/unsupervised/simcse" train.py \
    --model "$BERT" \
    --train_path "$ROOT/_minimal_data/simcse_train_s.txt" \
    --dev_path "$ROOT/_minimal_data/simcse_dev.tsv" \
    --save_dir checkpoints/LCQMC \
    --img_log_dir logs/LCQMC --img_log_name ESimCSE \
    --learning_rate 1e-5 --dropout 0.3 \
    --batch_size 16 --max_seq_len 64 --valid_steps 200 --logging_steps 50 \
    --num_train_epochs 1 --device cpu

run_task "RLHF_PPO" 5400 "$ROOT/RLHF" ppo_sentiment_example.py

echo
echo "=============================================================="
cat "$SUMMARY"
