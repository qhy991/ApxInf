"""Verify curated source references on CPU. No candidate code is imported or executed."""
from pathlib import Path
import ast
import hashlib
import json

def main():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / 'doc/qwen-image-metal-transfer/MANIFEST.json').read_text())
    for entry in manifest['candidates'] + manifest['documents']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Manifest path escapes this repository: ' + str(relative))
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Source hash mismatch: ' + str(relative))
        if relative.suffix == '.py':
            ast.parse(data, filename=str(relative))
    ids = {entry['id'] for entry in manifest['candidates']}
    if len(ids) != len(manifest['candidates']):
        raise ValueError('Candidate IDs are not unique')
    if manifest['runtime_source_changed'] or manifest['hardware_runs_started']:
        raise ValueError('Reference package unexpectedly claims runtime changes or hardware execution')
    print('PASS: ' + str(len(ids)) + ' candidate sources and ' + str(len(manifest['documents'])) + ' reference documents verified on CPU; no model execution.')

if __name__ == '__main__':
    main()
