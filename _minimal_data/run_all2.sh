#!/bin/bash
# 第二批：依赖 T5 / GPT2 的任务。模型已全部下载到本地 _models/，全程离线运行。
# 用法: bash _minimal_data/run_all2.sh

ROOT="D:/new plm"
PY="D:/Anaconda/envs/plm/python.exe"
T5="$ROOT/_models/t5-base-chinese-cluecorpussmall"
LOG="$ROOT/_minimal_data/runlogs"
mkdir -p "$LOG"

export HF_HUB_OFFLINE=1
export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

SUMMARY="$LOG/_summary2.tsv"
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
        note=$(grep -aE "best F1 performence has been updated|Evaluation" "$LOG/$name.log" | tail -1 | tr -d '\r')
        [ -z "$note" ] && note=$(grep -a "mean-reward" "$LOG/$name.log" | tail -1 | tr -d '\r')
    elif [ $code -eq 124 ]; then
        status="TIMEOUT"; note="超过 ${limit}s 被中断"
    else
        status="FAIL($code)"
        note=$(grep -aiE "error|Traceback" "$LOG/$name.log" | tail -1 | cut -c1-110 | tr -d '\r')
    fi

    printf "%-28s\t%-10s\t%-8s\t%s\n" "$name" "$status" "$dur" "$note" >> "$SUMMARY"
    printf "[%s] %-26s %8ss  %s\n" "$status" "$name" "$dur" "$note"
}

# 8. T5 生成式问答（采样 2000 条）
run_task "T5_QA" 4200 "$ROOT/answer_generation" train.py \
    --pretrained_model "$T5" \
    --save_dir checkpoints/DuReaderQG \
    --train_path "$ROOT/_minimal_data/qa_train.json" \
    --dev_path "$ROOT/_minimal_data/qa_dev.json" \
    --img_log_dir logs/DuReaderQG --img_log_name T5-QA \
    --batch_size 8 --max_source_seq_len 128 --max_target_seq_len 32 \
    --learning_rate 5e-5 --num_train_epochs 1 \
    --logging_steps 20 --valid_steps 100 --device cpu

# 9. T5 Mask-Then-Fill 填空模型（采样 5000 条）
run_task "T5_filling" 4200 "$ROOT/data_augment/filling_model" train.py \
    --pretrained_model "$T5" \
    --save_dir checkpoints/t5 \
    --train_path "$ROOT/_minimal_data/fill_train.tsv" \
    --dev_path "$ROOT/_minimal_data/fill_dev.tsv" \
    --img_log_dir logs --img_log_name T5-Filling \
    --batch_size 16 --max_source_seq_len 128 --max_target_seq_len 32 \
    --learning_rate 1e-4 --num_train_epochs 1 \
    --logging_steps 20 --valid_steps 100 --device cpu

# 10. RLHF PPO（已把轮数从 157 改为 5，模型指向本地目录）
run_task "RLHF_PPO" 4200 "$ROOT/RLHF" ppo_sentiment_example.py

echo
echo "=============================================================="
echo "第二批执行完毕，汇总："
echo "=============================================================="
cat "$SUMMARY"
