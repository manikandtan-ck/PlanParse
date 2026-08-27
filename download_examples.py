"""Download public construction-PDF examples into ignored examples/pdfs/."""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).parent
OUT = ROOT / "examples/pdfs"


def main() -> None:
    sources = json.loads((ROOT / "examples/public_examples.json").read_text())["sources"]
    OUT.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = OUT / source["local_filename"]
        request = urllib.request.Request(source["source_url"], headers={"User-Agent": "PlanParse PDF examples"})
        with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        print(f"downloaded {source['id']} -> {target}")


if __name__ == "__main__":
    main()
