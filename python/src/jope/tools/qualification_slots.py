"""Borrow up to sixteen of the existing GP cores without preempting searches."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import time

import experiment_runtime as runtime
from promotion_gate import immutable
from resume_expired_cycle import atomic, read, require, sha


CPUS = tuple(range(8,60)) + tuple(range(70,122))


def release_stale(root,manifest):
    """Transient-unit ExecStopPost: expiry never bypasses inherited slot flocks."""
    root=Path(root);manifest=Path(manifest);control=root/'runs/gp-qualification-reserve.json'
    with (root/'runs/gp-qualification-owner.lock').open('a') as owner:
        try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        if not control.exists():return False
        request=read(control)
        if request['manifest']!=str(manifest.relative_to(root)) or request['manifest_sha256']!=sha(manifest):return False
        atomic(control,dict(request,expires_at=request['created_at']+1))
        return True


@contextmanager
def acquire(root, manifest, output, count=16, seconds=7200, wait_seconds=660):
    root=Path(root);manifest=Path(manifest);output=Path(output)
    require(1 <= count <= 16 and 1 <= seconds <= 7200, 'qualification slot/lifetime limit exceeded')
    gp=read(root/'runs/gp-service.json')
    require(gp.get('slot_leases') is True and gp['workers']==104 and gp['threads_each']==1,
            'GP service does not support the qualified slot protocol')
    owner=(root/'runs/gp-qualification-owner.lock').open('a')
    leases=[];control=root/'runs/gp-qualification-reserve.json';request=None
    previous_fds=runtime._lease_fds
    try:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if control.exists():
            old=read(control)
            require(old['expires_at']<=time.time(),'another qualification reservation is active')
        now=int(time.time());slots=list(range(104-count,104))
        request=dict(schema='parallel.gp_slot_reservation.v1',manifest=str(manifest.relative_to(root)),
            manifest_sha256=sha(manifest),slots=slots,created_at=now,expires_at=now+seconds)
        output.mkdir(parents=True,exist_ok=True)
        immutable(output/'slot-reservation.json',request);atomic(control,request)
        deadline=min(time.monotonic()+wait_seconds,time.monotonic()+seconds)
        for slot in slots:
            lease=(root/f'runs/gp-slot-{slot}.lock').open('a');leases.append(lease)
            while True:
                require(read(control)==request and sha(manifest)==request['manifest_sha256'],
                        'qualification slot reservation changed while waiting')
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    require(time.monotonic()<deadline,'existing GP searches did not release their slots')
                    time.sleep(.1)
        runtime._lease_fds=(*previous_fds,*(lease.fileno() for lease in leases))
        immutable(output/'slots-acquired.json',dict(timestamp=time.time(),slots=slots,
            cpus=[CPUS[slot] for slot in slots],manifest_sha256=request['manifest_sha256']))
        yield tuple(CPUS[slot] for slot in slots)
    finally:
        runtime._lease_fds=previous_fds
        # Inherited copies remain locked until the caller's children exit.
        for lease in leases:lease.close()
        if request is not None and control.exists() and read(control)==request:
            atomic(control,dict(request,expires_at=request['created_at']+1))
        owner.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Release a dead qualification reservation after its unit exits.')
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    arguments=parser.parse_args()
    if arguments.manifest.exists():release_stale(arguments.root.resolve(),arguments.manifest.resolve())
