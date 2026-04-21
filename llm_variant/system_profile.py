from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_variant.config import detect_system_profile, recommended_ollama_model, recommended_openai_model


def main() -> None:
    profile = detect_system_profile()
    payload = {
        "os_name": profile.os_name,
        "machine": profile.machine,
        "cpu_count": profile.cpu_count,
        "ram_gb": profile.ram_gb,
        "recommended_ollama_model": recommended_ollama_model(profile),
        "recommended_openai_model": recommended_openai_model(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
