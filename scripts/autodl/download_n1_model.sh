#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source "${ENV_FILE:-.env}"; set +a
source .venv/bin/activate
: "${MODEL_ROOT:?}"; : "${HF_HOME:?}"
export HF_HOME
REPO_ID="Qwen/Qwen2.5-14B-Instruct"
REVISION="${QWEN14B_REVISION:-main}"
DEST="$MODEL_ROOT/Qwen2.5-14B-Instruct"
mkdir -p "$DEST"

python - "$REPO_ID" "$REVISION" "$DEST" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo, revision, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
info = HfApi().model_info(repo, revision=revision)
snapshot_download(repo_id=repo, revision=info.sha, local_dir=dest)
(dest / "model_revision.json").write_text(json.dumps({"repo_id": repo, "requested_revision": revision, "commit": info.sha}, indent=2))
PY

python - "$DEST" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for name in ("config.json", "tokenizer_config.json"):
    assert (root / name).is_file(), f"missing {name}"
assert (root / "tokenizer.json").is_file() or ((root / "vocab.json").is_file() and (root / "merges.txt").is_file())
index = root / "model.safetensors.index.json"
if index.is_file():
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
else:
    shards = [p.name for p in root.glob("*.safetensors")]
assert shards, "no safetensors weights found"
missing = [name for name in shards if not (root / name).is_file()]
assert not missing, f"missing weight shards: {missing}"
print(f"verified {len(shards)} weight shard(s)")
print((root / "model_revision.json").read_text())
PY
du -sh "$DEST"
