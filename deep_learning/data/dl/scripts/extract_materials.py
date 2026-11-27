"""JSON transport for the existing, unchanged LLC material extractor."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deep_learning.reference_backend.src.contracts import ContractError
from deep_learning.reference_backend.src.materials import extract_materials


def main():
    try:
        raw = sys.stdin.buffer.read(12 * 1024 * 1024 + 1)
        if len(raw) > 12 * 1024 * 1024:
            raise ContractError("Material request exceeds the transport limit.", "MATERIAL_TOO_LARGE")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {"attachments"}:
            raise ContractError("Expected an attachments object.", "MATERIAL_INVALID")
        result = extract_materials(payload["attachments"])
    except ContractError as error:
        print(json.dumps({"error": {"code": error.code, "message": str(error)}}, ensure_ascii=False))
        return 2
    except (ValueError, UnicodeError):
        print(json.dumps({"error": {"code": "MATERIAL_INVALID", "message": "Invalid material request."}}))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
