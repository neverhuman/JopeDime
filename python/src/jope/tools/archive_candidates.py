#!/usr/bin/env python3
"""Indexed historical candidate inventory; scores remain informational only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

from promotion_gate import immutable
from resume_expired_cycle import read, sha


def sources(bank_path, database):
    bank = read(bank_path)
    rows = []
    for index, row in enumerate(bank['tasks']['regression']):
        rows.append(dict(program=row['canonical_genome'], source=dict(kind='reviewed_seed_bank',
            path=str(bank_path), sha256=sha(bank_path), row=index,
            historical_information=row, selection_history=bank['selection'],
            selection_source=bank['source'], provenance_reasons=[
                'test-exact-selected bank; selection-family overlap has not been cleared',
                'per-program evaluator and selection receipt lineage unresolved'])))
    connection = sqlite3.connect('file:' + str(database.resolve()) + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        # Bounded winner table extraction; never scan the 98.9 GB strict corpus.
        winners = connection.execute("SELECT * FROM best_scores WHERE task='regression' ORDER BY dataset_hash,score_tier").fetchall()
        for row in winners:
            value = dict(row)
            program = value.pop('canonical_genome')
            if not program:
                found = connection.execute('SELECT canonical_genome FROM genomes WHERE semantic_sha256=?',
                                           (value['semantic_sha256'],)).fetchone()
                program = found[0] if found else ''
            reasons = ['historical selection family has not been mapped to the frozen current cohort',
                       'current evaluator refit required; historical scores are not labels']
            if not value['receipt_ref']:
                reasons.append('winner has no source evaluation receipt reference')
            if value['score_tier'] in ('test_exact', 'test_adaptive'):
                reasons.append('program selected using historical test targets')
            rows.append(dict(program=program, source=dict(kind='regression_winner', path=str(database),
                historical_information=value, evaluator_identity='unresolved in winner table',
                provenance_reasons=reasons)))
    finally:
        connection.close()
    return rows


def inventory(source_rows, compatibility, vocabulary_sha, executable_sha):
    audited = {r['input']:r for r in compatibility}
    programs = {}
    for row in source_rows:
        audit = audited[row['program']]
        canonical = audit['canonical_program'] or row['program']
        key = hashlib.sha256(canonical.encode()).hexdigest()
        entry = programs.setdefault(key, dict(program_sha256=key, canonical_program=canonical,
            exact_compatible=audit['exact_compatible'], compatibility_rejection=audit['rejection_reason'],
            admission_track='collection_only_diagnostic', transfer_eligible=False,
            current_evaluation_receipts=[], sources=[], provenance_reasons=[]))
        entry['sources'].append(row['source'])
        entry['provenance_reasons'] = sorted(set(entry['provenance_reasons'] + row['source']['provenance_reasons']))
    rows = [programs[k] for k in sorted(programs)]
    return dict(schema='parallel.archived_program_inventory.v1', vocabulary_sha256=vocabulary_sha,
        compatibility_executable_sha256=executable_sha, source_rows=len(source_rows), programs=rows,
        counts=dict(unique_programs=len(rows), exactly_compatible=sum(r['exact_compatible'] for r in rows),
                    transfer_eligible=0, collection_only=len(rows)),
        compatibility_rejections=dict(Counter(r['compatibility_rejection'] for r in rows if not r['exact_compatible'])),
        historical_scores_are_current_labels=False, translations_performed=False, final_tests_opened=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--genebank', type=Path, required=True)
    p.add_argument('--vocabulary', type=Path, required=True)
    p.add_argument('--compatibility-executable', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = sources(args.bank, args.genebank)
    immutable(args.output / 'sources.json', rows)
    immutable(args.output / 'programs.json', sorted({r['program'] for r in rows}))
    output = subprocess.check_output([str(args.compatibility_executable.resolve()), str(args.vocabulary.resolve()),
                                      str((args.output / 'programs.json').resolve())], text=True, timeout=120)
    compatibility = json.loads(output)
    immutable(args.output / 'compatibility.json', compatibility)
    result = inventory(rows, compatibility, sha(args.vocabulary), sha(args.compatibility_executable))
    immutable(args.output / 'inventory.json', result)
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
