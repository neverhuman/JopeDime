#!/usr/bin/env python3
"""Loopback-only HTTP server on xbabe2. No endpoint can control experiments."""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import json
from pathlib import Path
import os
import time
import csv
import io

HERE=Path(__file__).resolve().parent
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url=self.path.split('?')[0]
        files={'/DL_CONTINUAL_MODEL_V1.md':'DL_CONTINUAL_MODEL_V1.md','/DL_FLYWHEEL_CORRECTION_PLAN.md':'DL_FLYWHEEL_CORRECTION_PLAN.md','/DL_FLYWHEEL_KUMOS_ROSTER_AUDIT.md':'DL_FLYWHEEL_KUMOS_ROSTER_AUDIT.md','/DL_FLYWHEEL_EVERGREEN_IMPLEMENTATION.md':'DL_FLYWHEEL_EVERGREEN_IMPLEMENTATION.md','/DL_PAPER_S3_CUSTODY.md':'DL_PAPER_S3_CUSTODY.md','/DL_FLYWHEEL_ENCODER_PUBLICATION.md':'DL_FLYWHEEL_ENCODER_PUBLICATION.md','/api/wave4-full-comparison':'DL_wave4_full_comparison.json','/DL_wave4_full_comparison.md':'DL_wave4_full_comparison.md','/':'index.html','/paper/':'DL_paper_dashboard.html','/api/paper-evidence':'DL_paper_evidence_snapshot.json','/DL_WHITE_PAPER_TRACKING.md':'DL_WHITE_PAPER_TRACKING.md','/DL_FLYWHEEL_PLAN.md':'DL_FLYWHEEL_PLAN.md','/DL_FLYWHEEL_INTEGRATION_AUDIT.md':'DL_FLYWHEEL_INTEGRATION_AUDIT.md','/full/':'DL_wave4_full_dashboard.html','/api/wave4-full-data':'DL_wave4_full_data_snapshot.json','/DL_wave4_full_data_report.md':'DL_wave4_full_data_report.md','/DL_wave4_full_data_audit.md':'DL_wave4_full_data_audit.md','/app.js':'app.js','/style.css':'style.css','/api/metrics':'snapshot.json','/DL_wave4_v1.0.0.md':'DL_wave4_v1.0.0.md','/DL_wave4_full_report.md':'DL_wave4_full_report.md','/api/wave4-report':'DL_wave4_full_report.json'}
        filename=None
        if url in ['/export.html','/export.csv']:
            p=HERE/'snapshot.json'
            if not p.exists():self.send_error(503,'First snapshot pending');return
            snapshot=p.read_text()
            if url=='/export.html':
                html=(HERE/'index.html').read_text()
                html=html.replace('<link rel="stylesheet" href="/style.css">','<style>'+(HERE/'style.css').read_text()+'</style>')
                js=(HERE/'app.js').read_text().replace('</script','<\\/script')
                html=html.replace('<script src="/app.js"></script>','<script>window.__BOOTSTRAP__='+snapshot.replace('<','\\u003c')+';</script><script data-app>'+js+'</script>')
                b=html.encode();mime='text/html; charset=utf-8';filename='JopeDime_experiments_dashboard.html'
            else:
                d=json.loads(snapshot);buf=io.StringIO();writer=csv.writer(buf)
                writer.writerow(['run_id','architecture','seed','repeat_fold','epoch','train_loss','validation_loss','validation_r2','validation_auc','lr','grad_norm'])
                for host in d.get('hosts',[]):
                    for run in host.get('runs',[]):
                        for fold,history in run.get('history',{}).items():
                            for x in history:writer.writerow([run['id'],run['summary'].get('arch'),run['seed'],fold,x.get('ep'),x.get('train_loss'),x.get('val_loss'),x.get('val_r2'),x.get('val_auc'),x.get('lr'),x.get('gn')])
                b=buf.getvalue().encode();mime='text/csv; charset=utf-8';filename='JopeDime_training_metrics.csv'
        elif url=='/export-full.csv':
            p=HERE/'DL_wave4_full_data_snapshot.json'
            if not p.exists():self.send_error(503,'First full-data snapshot pending');return
            d=json.loads(p.read_text());buf=io.StringIO();writer=csv.writer(buf)
            writer.writerow(['campaign_id','manifest_sha256','bundle_sha256','source_sha','run_id','seed','repeat_fold','epoch_1based','optimizer_step','train_panel_fitness_mse','validation_row_fitness_mse','validation_equal_lineage_fitness_mse','optimizer_train_loss','validation_loss','fold_complete','selected_epoch_1based','snapshot_utc_unix'])
            for run in d.get('runs',[]):
                for fold,history in run.get('history',{}).items():
                    f=run['summary'].get('folds',{}).get(fold)
                    for x in history:writer.writerow([d['campaign_id'],d['manifest_sha256'],d['bundle_sha256'],d['source_sha'],run['job']['run_id'],run['job']['seed'],fold,x['ep']+1,x.get('step'),x.get('train_eval_fitness_mse'),x.get('val_eval_fitness_mse'),x.get('val_macro_fitness_mse'),x.get('train_loss'),x.get('val_loss'),f is not None,(f['best_ep']+1) if f else None,d['at']])
            b=buf.getvalue().encode();mime='text/csv; charset=utf-8';filename='DL_wave4_full_training_metrics.csv'
        elif url=='/api/health':
            p=HERE/'snapshot.json';b=json.dumps({'host':'xbabe2','pid':os.getpid(),'snapshot_age_seconds':time.time()-p.stat().st_mtime if p.exists() else None}).encode();mime='application/json'
        elif url in files:
            p=HERE/files[url]
            if not p.exists():self.send_error(503,'First snapshot pending');return
            b=p.read_bytes();mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json','.md':'text/markdown; charset=utf-8'}[p.suffix]
        else:self.send_error(404);return
        if url in ('/DL_CONTINUAL_MODEL_V1.md','/DL_FLYWHEEL_CORRECTION_PLAN.md','/DL_FLYWHEEL_KUMOS_ROSTER_AUDIT.md','/DL_FLYWHEEL_EVERGREEN_IMPLEMENTATION.md','/DL_PAPER_S3_CUSTODY.md','/DL_FLYWHEEL_ENCODER_PUBLICATION.md','/DL_wave4_full_comparison.md','/DL_WHITE_PAPER_TRACKING.md','/DL_FLYWHEEL_PLAN.md','/DL_FLYWHEEL_INTEGRATION_AUDIT.md','/DL_wave4_v1.0.0.md','/DL_wave4_full_report.md','/DL_wave4_full_data_report.md','/DL_wave4_full_data_audit.md'):filename=url.lstrip('/')
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        if filename:self.send_header('Content-Disposition','attachment; filename="'+filename+'"')
        self.end_headers();self.wfile.write(b)
    def log_message(self,*a):pass
if __name__=='__main__':
    print('http://127.0.0.1:8792/ — xbabe2; read-only snapshot service',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8792),Handler).serve_forever()
