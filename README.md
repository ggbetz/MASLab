# MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems

![MASLab](./assets/maslab_figure.png)

> [!NOTE]
> This fork adds support for MCP servers and async inference 


## Key Features
- **Comprehensive:** MASLab integrates over 20 LLM-based MAS methods (since March 2023)
- **Unified:** MASLab unified data pre-processing and evaluation protocols to ensure fair comparisons.
- **Research-Friendly:** MASLab implements methods within a shared streamlined structure.

## News
- [20250523] Release the preprint version! See the [ArXiv](https://arxiv.org/pdf/2505.16988).
- [20250420] Release the initial version! See the initial manuscript [here](./assets/MASLab_github.pdf).

## Get Started

1. Specify your model configs in `./model_api_configs/model_api_config.json`:

```json
"gpt-4o-mini-2024-07-18": {
        "model_list": [
            {"model_name": "gpt-4o-mini-2024-07-18", "model_url": "http://a.b.c.d:e/v1", "api_key": "xyz"}
        ],
        "max_workers_per_model": 10
    }
```

2. Install MCP server

```bash
python3.11 -m venv .venv-cedrus
source .venv-cedrus/bin/activate
pip install git+https://github.com/<cedrus-org>/cedrus
```

3. To see if the codebase is executable (e.g., vanilla, cot, agentverse)

```bash
uv run python inference.py --method_name <method_name> --debug
```

4. To inference on a dataset

```bash
# Step 1: build the test dataset
uv run python datasets/build_test_dataset.py --dataset_name <dataset_name>

# Step 2: inference on the whole dataset
uv run python inference.py \
  --test_dataset_name <dataset_name> \
  --method_name <inference_method_name> \
  --method_config_name <config_filenamebase> \
  --model_api_config <path_to_api_config> \
  --model_name <inference_model_name> \
```

Results will be stored in `results/{dataset_name}/{inference_model_name}/{inference_method_name}_infer.jsonl`.

5. To evaluate
```bash
uv run python evaluate.py \
  --tested_dataset_name <dataset_name> \
  --tested_method_name <inference_method_name> \
  --tested_method_config_name <config_filenamebase> \
  --tested_mas_model_name <inference_model_name> \
  --model_name <verifier_model_name> \
  --model_api_config <path_to_api_config> \
```

## Citation
```
@article{ye2025maslab,
  title={MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems},
  author={Ye, Rui and Huang, Keduan and Wu, Qimin and Cai, Yuzhu and Jin, Tian and Pang, Xianghe and Liu, Xiangrui and Su, Jiaqi and Qian, Chen and Tang, Bohan and others},
  journal={arXiv preprint arXiv:2505.16988},
  year={2025}
}
```
