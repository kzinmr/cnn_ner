python3 main.py \
--data_dir="/app/workspace/conll03/" \
--train_batch_size=128 \
--eval_batch_size=64 \
--num_workers=6 \
--learning_rate=0.001 \
--weight_decay=0.05 \
--anneal_factor=0.5 \
--patience=10 \
--max_epochs=150 \
--output_dir="/app/workspace/output/" \
--config_path="/app/workspace/models/train.config" \
--vocab_path="/app/workspace/models/" \
--monitor="f1" \
--monitor_training \
--do_train
#  \
# --gpus=1
# --download
# --do_predict
#--pretrain_embed_path="/app/workspace/models/word2vec.txt" \
#--model_path="/app/workspace/models/checkpoint-epoch=40-val_f1=0.60.ckpt" \
