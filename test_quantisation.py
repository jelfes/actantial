
import pandas as pd
from actantial.runner import run_extract
from actantial.backends import HuggingFaceBackend

backend = HuggingFaceBackend(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", quantisation=True)
data = pd.read_csv("test_data/test_data.csv")

run_extract(
    data=data.head(10),
    backend=backend,
    data_path="test_output",
    template="prompt_open_1"
)