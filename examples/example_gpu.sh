python -m actantial \
    --data_file "examples/data/sample_news_articles.csv" \
    --output_dir "examples/output" \
    --backend huggingface \
    --repository "google" \
    --model "gemma-3-1b-it" \
    --template "prompt_open"