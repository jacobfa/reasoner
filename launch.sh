#!/bin/bash

SESSION_NAME="training_session"
LOG_FILE="training_log.txt"

tmux new-session -d -s $SESSION_NAME

tmux send-keys -t $SESSION_NAME "
source ~/.bashrc  # Load environment variables, if necessary
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u train.py \
> $LOG_FILE 2>&1
" C-m

echo "Training started in tmux session '$SESSION_NAME'."
echo "Check '$LOG_FILE' for logs."
