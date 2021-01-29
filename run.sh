
python3 main.py \
--config_path="/app/workspace/cnn_model/train.config.cnn" \
--vocab_path="/app/workspace/cnn_model/" \
--nbest=1 \
--batch-size=32 \
--max-sent-length=250 \
--number-normalized=True \
--max_epochs=100 \
--data_dir="/app/workspace/data/" \
--output_dir="/app/workspace/output/" \
--do_train \
--pretrain_embed_path="/app/workspace/models/word2vec.txt" \
# --download
# --model_path="/app/workspace/cnn_model/cnn.0.model" \