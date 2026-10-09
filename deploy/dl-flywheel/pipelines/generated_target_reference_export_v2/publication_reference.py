"""Project only the reviewed existing publication envelope; never mutate it."""
from ..generated_target_admission_v1.common import need, reference

IDENTITY_KEYS = frozenset({'bucket', 'key', 'version_id', 'sha256', 'bytes'})
ANNOTATION_KEYS = frozenset({'state', 'verification'})


def publication_identity(ref):
    """Accept strict five-field or exact known create-only publication receipt."""
    need(type(ref) is dict, 'closed publication reference')
    keys = frozenset(ref)
    need(keys == IDENTITY_KEYS or keys == IDENTITY_KEYS | ANNOTATION_KEYS,
         'closed publication reference fields')
    if keys != IDENTITY_KEYS:
        need(ref['state'] == 'created' and
             ref['verification'] == 'exact VersionId GET + SHA256',
             'qualified publication receipt annotations')
    identity = {key: ref[key] for key in IDENTITY_KEYS}
    reference(identity)
    return identity
