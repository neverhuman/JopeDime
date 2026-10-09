"""Local-only one-use clock freeze after a pinned independent review."""
import argparse
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
GLOBAL = 1791592146.380285


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template-sha', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--review-sha', required=True)
    args = parser.parse_args()
    raw = (HERE / 'config.template.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.template_sha
    review_raw = Path(args.review).read_bytes()
    assert hashlib.sha256(review_raw).hexdigest() == args.review_sha
    review = json.loads(review_raw)
    assert review['state'] == 'passed' and review['template_sha256'] == args.template_sha
    config = json.loads(raw)
    assert config['created_unix'] is None and config['deadline_unix'] is None
    for name in ('source', 'base', 'prior_capture'):
        ref = config[name]; local = HERE / Path(ref['path']).name
        body = local.read_bytes()
        assert len(body) == ref['bytes'] and hashlib.sha256(body).hexdigest() == ref['sha256']
    now = time.time()
    assert now + 3600 <= GLOBAL == config['global_deadline_unix']
    config.update(created_unix=now, deadline_unix=now + 3600)
    target = HERE / 'config.frozen.json'
    body = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with target.open('xb') as stream: stream.write(body)
    print(json.dumps({'path': str(target), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}))


if __name__ == '__main__':
    main()
