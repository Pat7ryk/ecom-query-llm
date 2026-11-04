import json
from pathlib import Path

import pytest

from ecomq.data.audit_n1 import audit_datasets
from ecomq.eval.score_labels import bnb_4bit_kwargs


def _write(path: Path, split: str, query_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, query_id in enumerate(query_ids):
            row = {
                "id": f"{split}-{i}",
                "group": f"q{query_id}",
                "meta": {"query_id": query_id},
                "label": {"relevance": "exact"},
            }
            f.write(json.dumps(row) + "\n")


def test_audit_datasets_records_hashes_and_disjoint_queries(tmp_path):
    paths = {}
    for split, query_ids in {
        "train": [1, 2],
        "val": [3],
        "tune": [4, 5],
        "test": [6, 7],
    }.items():
        paths[split] = tmp_path / f"{split}.jsonl"
        _write(paths[split], split, query_ids)

    report = audit_datasets(paths)

    assert report["query_overlap"] == {}
    assert report["splits"]["train"]["rows"] == 2
    assert report["splits"]["train"]["queries"] == 2
    assert len(report["splits"]["train"]["sha256"]) == 64


def test_audit_datasets_rejects_query_leakage(tmp_path):
    paths = {}
    for split, query_ids in {
        "train": [1, 2],
        "val": [3],
        "tune": [2, 4],
        "test": [5],
    }.items():
        paths[split] = tmp_path / f"{split}.jsonl"
        _write(paths[split], split, query_ids)

    with pytest.raises(AssertionError, match="query leakage"):
        audit_datasets(paths)


def test_n1_config_preserves_e2a_except_model_and_explicit_defaults():
    yaml = Path("configs/sft/n1_qwen14b_qlora.yaml").read_text(encoding="utf-8")
    required = {
        "model_name_or_path": "Qwen/Qwen2.5-14B-Instruct",
        "quantization_bit": "4",
        "quantization_method": "bnb",
        "quantization_type": "nf4",
        "double_quantization": "true",
        "disable_gradient_checkpointing": "false",
        "lora_rank": "16",
        "lora_alpha": "32",
        "lora_dropout": "0.05",
        "lora_target": "all",
        "cutoff_len": "384",
        "per_device_train_batch_size": "16",
        "gradient_accumulation_steps": "2",
        "learning_rate": "1.0e-4",
        "num_train_epochs": "1.0",
        "load_best_model_at_end": "true",
        "metric_for_best_model": "eval_loss",
    }
    parsed = {}
    for line in yaml.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    assert {key: parsed.get(key) for key in required} == required


def test_label_scoring_can_match_training_double_quantization():
    assert bnb_4bit_kwargs(double_quant=True) == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bf16",
        "bnb_4bit_use_double_quant": True,
    }
