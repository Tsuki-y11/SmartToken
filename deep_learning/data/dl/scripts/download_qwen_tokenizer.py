"""Prepare only the fixed official tokenizer files used by the LLC Adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'artifacts/qwen-tokenizer-manifest.json').read_text(encoding='utf-8'))
    target = root/manifest['local_directory']
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest['files'].items():
        destination = target/name
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Existing tokenizer hash mismatch: {name}')
            continue
        if args.source_dir:
            source = args.source_dir/name
        else:
            from huggingface_hub import hf_hub_download
            source = Path(hf_hub_download(repo_id=manifest['repository'], filename=name, revision=manifest['revision']))
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source tokenizer hash mismatch: {name}')
        shutil.copyfile(source, destination)
    print(json.dumps(dict(verified_files=len(manifest['files']), directory=str(target))))


if __name__ == '__main__':
    main()
