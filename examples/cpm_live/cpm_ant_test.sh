export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 /root/zhaoyq/CPM-Bee/src/cpm_bee_test.py \
    --model-config /root/zhaoyq/models/10b/cpm-bee-10b.json \
    --load  /root/gongbt/cpm-bee-hf/models/pytorch_model.bin \
    --teacher-config /root/zhaoyq/models/10b/cpm-bee-10b.json \
    --load-teacher  /root/gongbt/cpm-bee-hf/models/pytorch_model.bin \
    --dataset /root/zhaoyq/CPM-Bee/tutorials/basic_task_finetune/bin_data/train \
    --cook-config /root/zhaoyq/BMCook/examples/cpm_live/configs/cpm-bee.json \
    --save-name /root/zhaoyq/models/cook/cooked_model.bin \
    --epoch 3 \
    --tensorboard /root/zhaoyq/tensorboard_log/bmcook \
    --batch-size 2 \
    --max-length 2048 \
    --lr 0.0001 \
    --warmup-iters 1 \
    --lr-decay-style noam \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --loss-scale 32768 \
    --start-step 0 \
