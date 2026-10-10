"""Download a pinned third-party MegaVul mirror and verify its LFS digest."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

URL = "https://huggingface.co/datasets/tronganhwork/CoVulPecker-Megavul/resolve/b76e218ee868b1b5efcd1074c7f2fc71f11a2eb5/megavul.json"
EXPECTED_SHA256 = "eabe8d3325be358d7a93327e3ae4f001725517a0067c4d80198f89904177419f"
EXPECTED_BYTES = 2668796982

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    started = time.perf_counter()
    digest = hashlib.sha256()
    count = 0
    with requests.get(URL, stream=True, timeout=(30, 60)) as response:
        response.raise_for_status()
        with partial.open("xb") as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    if count // (128 * 1024 * 1024) != (count-len(chunk)) // (128 * 1024 * 1024):
                        print(f"{count:,}/{EXPECTED_BYTES:,} bytes, {count/(time.perf_counter()-started)/1048576:.1f} MiB/s", flush=True)
    actual = digest.hexdigest()
    if count != EXPECTED_BYTES or actual != EXPECTED_SHA256:
        raise ValueError(f"Mirror integrity mismatch: bytes={count}, sha256={actual}")
    partial.replace(args.output)
    metadata = dict(url=URL, sha256=actual, bytes=count,
        provenance="third-party mirror, not byte-verified against the unavailable official OneDrive download",
        official_repository="https://github.com/Icyrockton/MegaVul",
        official_schema_revision="2b34eb3685b2a215d44c08d34fe9cec0f49eada3",
        mirror_revision="b76e218ee868b1b5efcd1074c7f2fc71f11a2eb5",
        acquired_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
    args.output.with_suffix(".provenance.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metadata), flush=True)

if __name__ == "__main__":
    main()
