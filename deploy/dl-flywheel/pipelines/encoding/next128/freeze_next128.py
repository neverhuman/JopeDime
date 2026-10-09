"""Local create-only shared clock freeze after an exact independent review.

No SSH, SQL, GPU or S3 action. Freeze and execution must use the same local host.
Do not run merely to prepare a proposal: freezing starts its sole1800s clock.
"""
import argparse
import copy
import json
import socket
import time
from pathlib import Path
import run_next128 as runner

HERE = Path(__file__).resolve().parent


def freeze(template_ref, review_ref, output, wall=None, monotonic=None):
    template = json.loads(runner.read(template_ref))
    runner.validate_template(template)
    review = json.loads(runner.read(review_ref))
    runner.need(review.get('state') == 'passed' and review.get('template_sha256') == template_ref['sha256'], 'independent review must bind exact template')
    runner.need(template['freezer']['sha256'] == runner.sha(Path(__file__).read_bytes()), 'freezer source changed')
    now = time.time() if wall is None else wall
    mono = time.monotonic() if monotonic is None else monotonic
    runner.need(now + 1800 <= runner.GLOBAL, 'full new window unavailable under original global ceiling')
    root = copy.deepcopy(template)
    root.update(approved_for_launch=True, created_unix=now, deadline_unix=now + 1800, shared_monotonic_deadline=mono + 1800, clock_host=socket.gethostname(), template=template_ref, review=review_ref)
    # This directory is an exclusive clock/attempt reservation. Any incomplete
    # freeze stays visible; it cannot be silently frozen again with a new clock.
    output.mkdir()
    runner.save(output / 'clock.json', {k: root[k] for k in ('created_unix', 'deadline_unix', 'shared_monotonic_deadline', 'clock_host', 'template', 'review')})
    for child in root['packets']:
        value = json.loads(runner.read(child['template']))
        value.update(approved_for_launch=True, review=review_ref)
        for key in ('created_unix', 'deadline_unix', 'shared_monotonic_deadline'):
            value[key] = root[key]
        child['config'] = runner.save(output / ('packet' + str(child['ordinal']) + '.json'), value)
    return runner.save(output / 'config.json', root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', required=True)
    parser.add_argument('--review-sha', required=True)
    parser.add_argument('--template-sha', required=True)
    args = parser.parse_args()
    template = runner.file_ref(HERE / 'config.template.json')
    review = runner.file_ref(Path(args.review).resolve())
    runner.need(template['sha256'] == args.template_sha and review['sha256'] == args.review_sha, 'explicit review/template pins differ')
    result = freeze(template, review, HERE / 'frozen')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
