"""Pure append-only contract, independent of any optimizer/admission API."""
from .common import digest, need


def validate_extension_transition(old, new):
    """Preserve every prior task and ordered ref; permit only unique tail refs."""
    need(set(old) == set(new) and all(old[k] == new[k] for k in old if k != 'tasks'),
         'extension contract changed')
    need(len(old['tasks']) == len(new['tasks']), 'old task removal or new context')
    for before, after in zip(old['tasks'], new['tasks']):
        need(set(before) == set(after) and
             all(before[k] == after[k] for k in before if k != 'program_results'), 'old task identity changed')
        a, b = before['program_results'], after['program_results']
        need(b[:len(a)] == a, 'old result reference removed/reordered/replaced')
        need(len({digest(r) for r in b}) == len(b), 'duplicate result weight')
    return True
