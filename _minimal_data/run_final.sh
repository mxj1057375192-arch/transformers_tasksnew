#!/bin/bash
# 最终队列：补救失败任务 + 剩余任务。顺序执行，每个任务带超时保护。
#
# 关键修正记录：
#   1. 文本匹配三个任务不能开 HF_HUB_OFFLINE —— evaluate 库要靠联网找指标脚本，
#      离线模式下会去本地找 ./accuracy/accuracy.py 然后失败。
#   2. UIE 已修 torch.load 的 map_location，纯 CPU 机器才能加载官方在 GPU 上存的权重。
#   3. RLHF reward 模型速度仅 0.04 step/s，数据从 3000 缩到 1000 条才能在超时内跑完。
#
# 用法: bash _minimal_data/run_final.sh

ROOT="D:/new plm"
PY="D:/Anaconda/envs/plm/python.exe"
BERT="$ROOT/_models/bert-base-chinese"
T5="$ROOT/_models/t5-base-chinese-cluecorpussmall"
LOG="$ROOT/_minimal_data/runlogs"
mkdir -p "$LOG"

export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

SUMMARY="$LOG/_summary3.tsv"
: > "$SUMMARY"
printf "%-28s\t%-10s\t%-8s\t%s\n" "TASK" "STATUS" "SECONDS" "NOTE" >> "$SUMMARY"

run_task() {
    local name="$1"; local limit="$2"; local offline="$3"; local dir="$4"; shift 4
    local start=$(date +%s)
    echo "=============================================================="
    echo "[START] $name   (超时上限 ${limit}s, 离线=${offline})"
    echo "=============================================================="
    if [ "$offline" = "yes" ]; then
        ( cd "$dir" && HF_HUB_OFFLINE=1 timeout "$limit" "$PY" -u "$@" ) > "$LOG/$name.log" 2>&1
    else
        ( cd "$dir" && HF_ENDPOINT=https://aifasthub.com timeout "$limit" "$PY" -u "$@" ) > "$LOG/$name.log" 2>&1
    fi
    local code=$?
    local dur=$(( $(date +%s) - start ))

    local status note
    if [ $code -eq 0 ]; then
        status="OK"
        note=$(grep -aE "best F1 performence has been updated|Evaluation|BLEU|mean-reward" "$LOG/$name.log" | tail -1 | tr -d '\r')
    elif [ $code -eq 124 ]; then
        status="TIMEOUT"; note="超过 ${limit}s 被中断"
    else
        status="FAIL($code)"
        note=$(grep -aiE "error|Traceback" "$LOG/$name.log" | tail -1 | cut -c1-110 | tr -d '\r')
    fi

    printf "%-28s\t%-10s\t%-8s\t%s\n" "$name" "$status" "$dur" "$note" >> "$SUMMARY"
    printf "[%s] %-26s %8ss  %s\n" "$status" "$name" "$dur" "$note"
}

# ---- 文本匹配三个（不设离线：evaluate 要联网找指标脚本）----
run_task "matching_pointwise" 2400 no "$ROOT/text_matching/supervised" train_pointwise.py \
    --model "$BERT" \
    --train_path data/comment_classify/train.txt --dev_path data/comment_classify/dev.txt \
    --save_dir checkpoints/comment_classify/pointwise \
    --img_log_dir logs/comment_classify --img_log_name PointWise \
    --batch_size 8 --max_seq_len 128 --valid_steps 100 --logging_steps 20 \
    --num_train_epochs 2 --device cpu

run_task "matching_dssm" 2400 no "$ROOT/text_matching/supervised" train_dssm.py \
    --model "$BERT" \
    --train_path data/comment_classify/train.txt --dev_path data/comment_classify/dev.txt \
    --save_dir checkpoints/comment_classify/dssm \
    --img_log_dir logs/comment_classify --img_log_name DSSM \
    --batch_size 8 --max_seq_len 128 --valid_steps 100 --logging_steps 20 \
    --num_train_epochs 2 --device cpu

run_task "matching_senttrans" 2400 no "$ROOT/text_matching/supervised" train_sentence_transformer.py \
    --model "$BERT" \
    --train_path data/comment_classify/train.txt --dev_path data/comment_classify/dev.txt \
    --save_dir checkpoints/comment_classify/sentence_transformer \
    --img_log_dir logs/comment_classify --img_log_name SentenceTransformer \
    --batch_size 8 --max_seq_len 128 --valid_steps 100 --logging_steps 20 \
    --num_train_epochs 2 --device cpu

# ---- UIE（已修 map_location）----
run_task "UIE" 3600 yes "$ROOT/UIE" train.py \
    --pretrained_model uie-base-zh \
    --save_dir checkpoints/DuIE \
    --train_path data/DuIE/train.txt --dev_path data/DuIE/dev.txt \
    --img_log_dir logs --img_log_name "UIE Base" \
    --batch_size 16 --max_seq_len 128 --learning_rate 5e-5 \
    --num_train_epochs 2 --logging_steps 20 --valid_steps 50 --device cpu

# ---- RLHF 奖励模型（1000 条；原速 0.04 step/s，3000 条会超时）----
run_task "RLHF_reward" 3600 yes "$ROOT/RLHF" train_reward_model.py \
    --model "$BERT" \
    --train_path "$ROOT/_minimal_data/rlhf_reward_train_s.tsv" \
    --dev_path "$ROOT/_minimal_data/rlhf_reward_dev_s.tsv" \
    --save_dir checkpoints/reward_model/sentiment_analysis \
    --img_log_dir logs/reward_model/sentiment_analysis --img_log_name "Reward Model" \
    --batch_size 16 --max_seq_len 128 --learning_rate 1e-5 \
    --valid_steps 40 --logging_steps 10 --num_train_epochs 1 --device cpu

# ---- SimCSE 无监督（1 万条）----
run_task "SimCSE" 5400 yes "$ROOT/text_matching/unsupervised/simcse" train.py \
    --model "$BERT" \
    --train_path "$ROOT/_minimal_data/simcse_train_s.txt" \
    --dev_path "$ROOT/_minimal_data/simcse_dev.tsv" \
    --save_dir checkpoints/LCQMC \
    --img_log_dir logs/LCQMC --img_log_name ESimCSE \
    --learning_rate 1e-5 --dropout 0.3 \
    --batch_size 16 --max_seq_len 64 --valid_steps 200 --logging_steps 50 \
    --num_train_epochs 1 --device cpu

# ---- T5 问答（1000 条）----
run_task "T5_QA" 4200 yes "$ROOT/answer_generation" train.py \
    --pretrained_model "$T5" \
    --save_dir checkpoints/DuReaderQG \
    --train_path "$ROOT/_minimal_data/qa_train_s.json" \
    --dev_path "$ROOT/_minimal_data/qa_dev.json" \
    --img_log_dir logs/DuReaderQG --img_log_name T5-QA \
    --batch_size 8 --max_source_seq_len 128 --max_target_seq_len 32 \
    --learning_rate 5e-5 --num_train_epochs 1 \
    --logging_steps 20 --valid_steps 100 --device cpu

# ---- T5 填空（2000 条）----
run_task "T5_filling" 4200 yes "$ROOT/data_augment/filling_model" train.py \
    --pretrained_model "$T5" \
    --save_dir checkpoints/t5 \
    --train_path "$ROOT/_minimal_data/fill_train_s.tsv" \
    --dev_path "$ROOT/_minimal_data/fill_dev.tsv" \
    --img_log_dir logs --img_log_name T5-Filling \
    --batch_size 16 --max_source_seq_len 128 --max_target_seq_len 32 \
    --learning_rate 1e-4 --num_train_epochs 1 \
    --logging_steps 20 --valid_steps 100 --device cpu

# ---- RLHF PPO（轮数已从 157 改为 5）----
run_task "RLHF_PPO" 4200 yes "$ROOT/RLHF" ppo_sentiment_example.py

echo
echo "=============================================================="
echo "最终队列执行完毕，汇总："
echo "=============================================================="
cat "$SUMMARY"
