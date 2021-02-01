
python3 main.py \
--data_dir="/app/workspace/data/" \
--config_path="/app/workspace/models/train.config" \
--vocab_path="/app/workspace/models/" \
--pretrain_embed_path="/app/workspace/models/word2vec.txt" \
--train_batch_size=128 \
--eval_batch_size=64 \
--num_workers=8 \
--max-sent-length=250 \
--number-normalized=True \
--max_epochs=100 \
--output_dir="/app/workspace/output/" \
--model_path="/app/workspace/cnn_pl_model/checkpoint-epoch=41-val_loss=224.52.ckpt" \
--do_train \
--gpus=1
# --download
# --do_predict