"""Download/copy exactly the frozen encoder files and verify every hash."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(target, manifest, source=None):
    if target.exists():
        expected = {row['path'] for row in manifest['files']}
        actual = {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}
        if expected != actual or any(sha(target / row['path']) != row['sha256'] for row in manifest['files']):
            raise ValueError('已有 Encoder 与冻结清单不符；请指定新的空目录，不会自动覆盖。')
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='encoder-staging-', dir=target.parent) as temp:
        staging = Path(temp) / 'files'
        for row in manifest['files']:
            if source is None:
                from huggingface_hub import hf_hub_download
                origin = Path(hf_hub_download(repo_id=manifest['repo_id'], revision=manifest['revision'], filename=row['path']))
            else:
                origin = source / row['path']
            if not origin.is_file() or origin.stat().st_size != row['size'] or sha(origin) != row['sha256']:
                raise ValueError('Encoder 文件校验失败：' + row['path'])
            destination = staging / row['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
        staging.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, help='已有冻结 Encoder 目录；不指定则从固定 revision 下载')
    parser.add_argument('--target-dir', type=Path, default=ROOT / 'artifacts/encoder')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'artifacts/encoder-manifest.json').read_text(encoding='utf-8'))
    target = prepare(args.target_dir.resolve(), manifest, args.source_dir)
    print(json.dumps({'encoder_path': str(target), 'verified_files': len(manifest['files'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
