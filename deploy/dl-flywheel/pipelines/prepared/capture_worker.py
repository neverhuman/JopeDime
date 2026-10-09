"""Read-only known-unstarted capture; writes only its own sealed receipts."""
import json, sys, time, resource
from pathlib import Path
from jope_flywheel.program_io import bound, read, sha, need, create
from jope_flywheel.protocol import canonical
from cold_reservation import prove
from reconcile import capture


def main():
    root=Path(__file__).resolve().parent
    raw=read(root/'capture-config.json');config=json.loads(raw)
    need(len(sys.argv)==3 and sys.argv[1]=='--config-sha'and sha(raw)==sys.argv[2], 'immutable capture packet')
    for name,pin in config['sources'].items():need(sha(read(root/name))==pin,'exact reviewed recovery source')
    campaign_raw=bound(config['campaign_config']);campaign=json.loads(campaign_raw)
    need(time.time()<config['deadline_unix']<=campaign['global_deadline_unix']==1791592146.380285,'fixed CPU capture deadline')
    started=time.time_ns();before=resource.getrusage(resource.RUSAGE_SELF)
    certified=prove(Path(campaign['owned_root'])/'training',config['cold_reservation_permit'])
    receipt=capture(Path(campaign['owned_root'])/'training',campaign_sha256=sha(campaign_raw),frozen=False,
        max_updates=campaign['neural']['max_updates'],plateau_patience=campaign['neural']['plateau_patience'],
        learning_rate_floor=campaign['neural']['learning_rate_floor'],require_closed_epoch=False,certified_unstarted=certified)
    need(receipt['absolute_step']==3297 and receipt['certified_batch']['batch_id']==certified['batch_id'], 'actual3297 with same certified3298 reservation')
    create(root/'cold-certificate.json',canonical(certified));create(root/'transfer-capture.json',canonical(receipt))
    after=resource.getrusage(resource.RUSAGE_SELF)
    create(root/'capture-status.json',canonical({'state':'passed','config_sha256':sha(raw),'started_utc_ns':started,'finished_utc_ns':time.time_ns(),
        'cpu_user_seconds':after.ru_utime-before.ru_utime,'cpu_system_seconds':after.ru_stime-before.ru_stime,
        'optimizer_updates':0,'planner_mutations':0,'outbox_mutations':0,'source_sha256':sha(read(Path(__file__))),
        'capture_sha256':sha(read(root/'transfer-capture.json')),'certificate_sha256':sha(read(root/'cold-certificate.json'))}))


if __name__=='__main__':main()
