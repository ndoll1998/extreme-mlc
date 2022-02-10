python mil/preprocess.py \
    --train-csv datasets/eurlex57k/train.csv \
    --val-csv datasets/eurlex57k/dev.csv \
    --test-csv datasets/eurlex57k/test.csv \
    --tokenizer-type sentence-transformers \
    --tokenizer-name all-MiniLM-L6-v2 \
    --max-seq-length 128 \
    --max-inst-count 4 \
    --output-dir data
