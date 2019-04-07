export GLUE_DIR=glue_data/SST-2
#export GLUE_DIR=SST2
export TASK_NAME=SST-2


python run_classifier.py \
       --task_name $TASK_NAME \
       --do_train \
       --do_eval \
       --do_lower_case \
       --data_dir $GLUE_DIR \
       --bert_model bert-base-uncased \
       --max_seq_length 128 \
       --train_batch_size 16 \
       --learning_rate 2e-5 \
       --num_train_epochs 3.0 \
       --output_dir output/$TASK_NAME/ \
       --cache_dir cache
