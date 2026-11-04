                engine.kill();assert engine.wait(timeout=5)==-9
                observer.close()
            engine,logpath=traced_engine(h,label+'-selected',{'JOPE_STORE_TRACE_BOUNDARIES':'1','JOPE_STORE_FAILPOINT':hook,'JOPE_STORE_FAILPOINT_OCCURRENCE':'1','JOPE_STORE_FAILPOINT_PARK':'1'})
            deadline=time.monotonic()+20
            while f'JOPE_STORE_CRASH:{hook}' not in logpath.read_text().splitlines():
                assert engine.poll() is None,'engine did not reach raw commit'
                assert time.monotonic()<deadline,'raw commit hook deadline'
                time.sleep(.01)
            assert f'JOPE_STORE_BOUNDARY:{hook}:1' in logpath.read_text().splitlines()
            engine.kill();signal_exit=engine.wait(timeout=5);assert signal_exit==-9
            recovery_homes.append(h);client=MCP(h)
            info=client.call('tools/call',{'name':'system_info','arguments':{}});assert not info.get('isError'),info
            inventory=run(h,['runs','list'])['runs']
            if phase=='initialization':
                assert not inventory
                d=run(h,['dataset','add',str(csv),'--name','int02-real-train','--idempotency-key','int02-register','--json'])['dataset_id']
                rid=run(h,spec_argv(d,'int02-boundary-run',2))['run_id']
                recovered=None;state='not-created';prefix=[];expected_best=snapshot['best']
            else:
                assert len(inventory)==1 and inventory[0]['run_id']==rid
                recovered=run(h,['status',rid,'--json']);assert recovered['state']=='paused'
                assert recovered==run(h,['status',rid,'--json']),'raw recovery was not idempotent'
                prefix=client.events(rid,0)['events'];assert prefix[:len(original_prefix)]==original_prefix
                assert project(prefix)==project(json.loads((out/'live-mcp-events.json').read_text()))[:len(project(prefix))]
                run(h,['control',rid,'--idempotency-key','int02-raw-continue','continue'])
                state='paused';expected_best=report['r4_final']['best']
            completed=wait_terminal(h,rid);events=client.events(rid,0)['events']
            assert completed['state']=='completed' and completed['best']==expected_best
            assert events[:len(prefix)]==prefix and [int(e['sequence']) for e in events]==list(range(1,len(events)+1))
            assert events[-1]['body']['nondeterminism']['count']=='0'
            assert not list((h/'tmp').iterdir())
            cases.append({'phase':phase,'hook':hook,'ordinal':1,'signal':signal_exit,'state':state,'events':len(prefix),'checkpoint':recovered,'completed':completed,'staging_clean':True,'continued_sequences':len(events),'reference_integrity_checked_on_open':True})
            save('engine-boundary-cases.json',cases)
            client.close()
            subprocess.run([args.binary,'serve','--stop'],env={**base_env,'JOPEDIME_HOME':str(h)},capture_output=True,text=True,timeout=10,check=True)
            check_descriptors()
    for hook in hooks:
        for ordinal in seen[hook]:
            label=f'b-{hooks.index(hook)}-{ordinal}'
            h,d=start_engine(label);seed=children[-1];stop_engine(h,seed)
            child,logpath=traced_engine(h,label,{'JOPE_STORE_TRACE_BOUNDARIES':'1','JOPE_STORE_FAILPOINT':hook,'JOPE_STORE_FAILPOINT_OCCURRENCE':str(ordinal),'JOPE_STORE_FAILPOINT_PARK':'1'})
            request=None;deadline=time.monotonic()+25
            while True:
                text=logpath.read_text()
                if f'JOPE_STORE_CRASH:{hook}' in text.splitlines():break
                assert child.poll() is None,f'{label}: engine exited before selected boundary: {text}'
                if request is None and daemon_ready(h,child):
                    with (out/(label+'-request.log')).open('w') as output:
                        request=subprocess.Popen([args.binary,*spec_argv(d,'int02-boundary-run',2)],env={**base_env,'JOPEDIME_HOME':str(h)},stdout=output,stderr=output)
                    children.append(request)
                assert time.monotonic()<deadline,f'{label}: real boundary not reached'
                time.sleep(.01)
            assert f'JOPE_STORE_BOUNDARY:{hook}:{ordinal}' in text.splitlines()
            child.kill();signal_exit=child.wait(timeout=5);assert signal_exit==-9
            if request is not None:
                try:request.wait(timeout=5)
                except subprocess.TimeoutExpired:request.terminate();request.wait(timeout=5)
            recovery_homes.append(h)
            client=MCP(h)
            # Invoke an actual tool: stdio initialization alone may defer engine discovery.
            info=client.call('tools/call',{'name':'system_info','arguments':{}})
            assert not info.get('isError'),info
            inventory=run(h,['runs','list'])['runs'];assert len(inventory)<=1
            recovered_events=[];state='not-created';recovered=None
            if inventory:
                rid=inventory[0]['run_id'];recovered=run(h,['status',rid,'--json'])
                state=recovered['state'];assert state in ['paused','completed'],(label,state)
                assert run(h,['status',rid,'--json'])==recovered,'recovery was not idempotent'
                recovered_events=client.events(rid,0)['events']
                assert [int(event['sequence']) for event in recovered_events]==list(range(1,len(recovered_events)+1))
                assert project(recovered_events)==expected[:len(project(recovered_events))],label
            assert not list((h/'tmp').iterdir()),f'{label}: staging survived automatic recovery'
            # Recovery verifies every durable record's CAS references before serving.
            # Execute the authorized continuation as well as proving the recovered prefix.
            if not inventory:
                rid=run(h,spec_argv(d,'int02-boundary-run',2))['run_id']
            elif state=='paused':
                control=['control',rid,'--idempotency-key','int02-boundary-continue','continue']
                resumed=run(h,control)
                assert run(h,control)==resumed,'Continue retry changed its committed outcome'
                changed=out/(label+'-changed-control.json')
                changed.write_text(json.dumps({'action':'continue','expected_revision':'0'}))
                conflict=subprocess.run([args.binary,'control',rid,'--idempotency-key','int02-boundary-continue','--body',str(changed),'continue'],env={**base_env,'JOPEDIME_HOME':str(h)},text=True,capture_output=True,timeout=10)
                commands.append({'argv':['control',rid,'changed-body-negative'],'exit':conflict.returncode,'stderr':conflict.stderr})
                assert conflict.returncode==8,'changed Continue request did not return the conflict exit'
                record=json.loads((h/'daemon.json').read_text());address=urllib.parse.urlsplit(record['url'])
                connection=http.client.HTTPConnection('127.0.0.1',address.port,timeout=10)
                try:
                    connection.request('POST',f'/v1/runs/{rid}/control',body=changed.read_text(),headers={'Authorization':'Bearer '+record['token'],'Content-Type':'application/json','Idempotency-Key':'int02-boundary-continue'})
                    reply=connection.getresponse();problem=json.loads(reply.read())
                    assert reply.status==409 and problem['code']=='IDEMPOTENCY_CONFLICT','changed Continue API request did not return its typed conflict'
                    save(label+'-conflict.json',{'status':reply.status,'problem':problem,'cli_exit':conflict.returncode})
                finally:connection.close()
                # A null is part of command identity even when its semantic default is absent.
                connection=http.client.HTTPConnection('127.0.0.1',address.port,timeout=10)
                try:
                    connection.request('POST',f'/v1/runs/{rid}/control',body=json.dumps({'action':'continue','expected_revision':None}),headers={'Authorization':'Bearer '+record['token'],'Content-Type':'application/json','Idempotency-Key':'int02-boundary-continue'})
                    reply=connection.getresponse();problem=json.loads(reply.read())
                    assert reply.status==409 and problem['code']=='IDEMPOTENCY_CONFLICT','explicit null was lost from Continue identity'
                finally:connection.close()
            completed=wait_terminal(h,rid)
            final_events=client.events(rid,0)['events']
            assert completed['state']=='completed' and completed['best']==snapshot['best'],(label,completed)
            assert final_events[:len(recovered_events)]==recovered_events,'continuation changed durable prefix'
            assert [int(event['sequence']) for event in final_events]==list(range(1,len(final_events)+1))
            assert final_events[-1]['body']['nondeterminism']['count']=='0','continuation hit a resource limit'
            cases.append({'hook':hook,'ordinal':ordinal,'signal':signal_exit,'state':state,'events':len(recovered_events),'staging_clean':True,'checkpoint':recovered,'completed':completed,'continued_sequences':len(final_events),'reference_integrity_checked_on_open':True})
            save('engine-boundary-cases.json',cases)
            client.close()
            subprocess.run([args.binary,'serve','--stop'],env={**base_env,'JOPEDIME_HOME':str(h)},capture_output=True,text=True,timeout=10,check=True)
            check_descriptors()
    assert len(cases)==sum(map(len,seen.values()))+4 and any(c['state']=='paused' for c in cases) and any(c['state']=='completed' for c in cases)
    normal_count=len(cases)-4
    assert len(cases)==normal_count+4
    report['r1_run_boundary_cases']=normal_count;report['r1_initialization_and_recovery_cases']=4
    report['r1_engine_boundary_cases']=len(cases);report['r1_engine_boundary_hooks']=hooks+['before_schema_commit','after_schema_commit','before_recovery_commit','after_recovery_commit']
    assert {case['hook'] for case in cases}==set(report['r1_engine_boundary_hooks'])
    report['r1_all_internal_background_cuts_qualified']=True

try:
    control=[{'run_id':'run_a','time':'a','sequence':'1','kind':'completed','body':{'experiment_id':'e1','model_id':'m1','budget_used':{'wall_seconds':'1','evaluations':'2'},'warnings':[{'host':True,'message':'host'},{'host':False,'message':'semantic'}]}}]
    operational=json.loads(json.dumps(control));operational[0]['run_id']='run_b';operational[0]['time']='b'
    operational[0]['body']['budget_used']['wall_seconds']='99';operational[0]['body']['warnings'][0]['message']='different host'
    assert project(control)==project(operational),'approved operational exclusions changed projection'
    for path in [('sequence',),('body','model_id'),('body','experiment_id'),('body','budget_used','evaluations')]:
        changed=json.loads(json.dumps(control));node=changed[0]
        for key in path[:-1]:node=node[key]
        node[path[-1]]='different'
        assert project(control)!=project(changed),f'projection hid semantic change: {path}'
    report['projection_oracle_controls']=True
    version=run(runtime,['--version','--json']);report['version']=version
    assert version['source_sha40']==args.source, 'binary source pin missing/mismatched'
    if not args.s2_only and not args.r1_only:
        home,dataset=start_engine('live')
        if args.capture_sse:
            from observe_http import Observer
            observers.append(Observer(home))
        mcp=MCP(home)
        started=run(home,spec_argv(dataset,'int02-r4-live',32));rid=started['run_id']
        report['r4_start_state']=started['state']
        report['terminal_first']=attach(home,rid,'terminal-before-close')
        midway=run(home,['status',rid,'--json']);report['state_at_close']=midway['state']
        first=mcp.events(rid,0,2);cursor=int(first['next_after']);events=list(first['events'])
        final=wait_terminal(home,rid);report['r4_final']=final
        report['terminal_reattach']=attach(home,rid,'terminal-after-reattach')
        while cursor<int(final['event_watermark']):
            page=mcp.events(rid,cursor,2);assert int(page['next_after'])>cursor
            cursor=int(page['next_after']);events.extend(page['events'])
        cli=subprocess.run([args.binary,'events',rid,'--after','0'],env={**base_env,'JOPEDIME_HOME':str(home)},
            text=True,capture_output=True,timeout=15)
        commands.append({'argv':['events',rid,'--after','0'],'exit':cli.returncode,'stdout':cli.stdout,'stderr':cli.stderr})
        cli_events=[json.loads(x) for x in cli.stdout.splitlines() if x.strip()] if cli.returncode==0 else []
        save('live-mcp-events.json',events);save('live-cli-events.json',cli_events);save('mcp-frames.json',mcp.frames)
        report['mcp_sequences_contiguous']=[int(x['sequence']) for x in events]==list(range(1,cursor+1))
        report['mcp_equals_cli_events']=events==cli_events
        report['terminal_event_projection_observed']=False
        report['r4_qualified']=False
        if args.capture_sse:
            traces=observers[-1].streams
            save('terminal-sse-streams.json',traces)
            equal=[trace for trace in traces if trace['events']==events]
            screen=(out/'terminal-after-reattach.pty').read_text(errors='replace')
            champion=final.get('best') or {}
            rendered=champion.get('algorithm_id','__missing__') in screen and f"{champion.get('cv_value',float('nan')):.3f}" in screen
            report['terminal_sse_stream_count']=len(traces)
            report['terminal_event_projection_observed']=bool(equal)
            report['terminal_full_stream_events']=len(equal[-1]['events']) if equal else 0
            report['terminal_champion_render_matches_mcp_snapshot']=rendered
            report['r4_qualified']=bool(equal) and rendered and report['mcp_sequences_contiguous'] and report['mcp_equals_cli_events'] and report['state_at_close']=='running' and report['terminal_first']['exit']==report['terminal_reattach']['exit']==0
            assert report['r4_qualified'],'live terminal wire projection/reattach acceptance failed'
        mcp.close()

    if not args.r4_only and not args.r1_only:
        # Rerun twice in fresh engine homes with identical registered train bytes/budgets/seed.
        pairs=[]
        for index in [1,2]:
            if not args.same_store or index==1:h,d=start_engine('rerun'+str(index))
            key='int02-s2-distinct-'+str(index) if args.same_store else 'int02-s2-same-key'
            argv=spec_argv(d,key,args.s2_evaluations)
            plan_argv=[x for x in argv if x!='--detach']+['--dry-run']
            plan_argv[plan_argv.index('--idempotency-key')+1]+='-plan'
            plan=run(h,plan_argv);save(f's2-{index}-plan.json',plan)
            start=run(h,argv)
            snap=wait_terminal(h,start['run_id']);c=MCP(h)
            page=c.events(start['run_id'],0,2);ev=list(page['events']);cursor=int(page['next_after'])
            assert c.close()==0,'MCP EOF did not close first observer'
            resumed=MCP(h)
            while cursor<int(snap['event_watermark']):
                page=resumed.events(start['run_id'],cursor,2)
                assert int(page['next_after'])>cursor,'MCP reconnect cursor did not advance'
                cursor=int(page['next_after']);ev.extend(page['events'])
            assert ev==resumed.events(start['run_id'],0)['events'],'MCP reconnect changed the committed journal'
            assert [int(event['sequence']) for event in ev]==list(range(1,len(ev)+1))
            save(f's2-{index}-reconnect-wire.json',{'before_disconnect':c.frames,'after_reconnect':resumed.frames})
            save(f's2-{index}-events.json',ev);save(f's2-{index}-snapshot.json',snap)
            pairs.append((snap,ev))
            resumed.close()
        assert pairs[0][0]['run_id']!=pairs[1][0]['run_id'],'must compare distinct logical runs, not an idempotent retry'
        report['distinct_run_ids']=True
        report['same_store']=args.same_store
        left,right=map(lambda pair:project(pair[1]),pairs)
        report['s2_event_counts']=[len(x[1]) for x in pairs]
        report['s2_projection_equal_without_run_id_time']=left==right
        report['s2_best_equal']=pairs[0][0].get('best')==pairs[1][0].get('best')
        report['s2_nondeterminism_counts']=[int(pair[1][-1]['body']['nondeterminism']['count']) for pair in pairs]
        report['s2_qualified']=left==right and report['s2_best_equal'] and report['s2_nondeterminism_counts']==[0,0]
        if args.assert_model_identity:
            assert report['s2_best_equal'] and report['s2_nondeterminism_counts']==[0,0], 'canonical model identity regression'
            first_plan=json.loads((out/'s2-1-plan.json').read_text());second_plan=json.loads((out/'s2-2-plan.json').read_text())
            assert first_plan['plan_digest']==second_plan['plan_digest'], 'different plans cannot qualify model determinism'
            report['canonical_model_identity_pass']=True
        save('s2-projections.json',{'first':left,'second':right})
        report['s2_mcp_process_reconnects']=2
        report['projection_ruling']='rulings_t1900:R2'
        if args.assert_s2:assert report['s2_qualified'],'R2 S2 projection/champion mismatch'
    if args.r1_only or args.all_live:
        h,d=start_engine('crash')
        engine=children[-1]
        started=run(h,spec_argv(d,'int02-r1-native',32));rid=started['run_id']
        deadline=time.monotonic()+20
        while True:
            before=run(h,['status',rid,'--json'])
            assert before['state']=='running','run ended before selected live cut'
            if before.get('best') is not None and int(before['event_watermark'])>=5:break
            assert time.monotonic()<deadline,'no committed first model before live cut'
            time.sleep(.02)
        client=MCP(h);prefix=client.events(rid,0)['events']
        engine.kill();signal_exit=engine.wait(timeout=5)
        assert signal_exit==-9,'must be actual own background engine SIGKILL'
        client.p.terminate();client.close()
        recovery_homes.append(h)
        report['r1_signal_exit']=signal_exit
        save('r1-before.json',before);save('r1-prefix.json',prefix);save('report-intermediate.json',report)
        report['r1_next_interactive_attach']=attach(h,rid,'terminal-next-use')
        assert report['r1_next_interactive_attach']['exit']==0,'interactive next-use failed to recover/start engine'
        recovered=run(h,['status',rid,'--json'])
        repeated=run(h,['status',rid,'--json'])
        assert recovered['state']=='paused' and recovered.get('best') is not None
        assert int(recovered['revision'])>=int(before['revision'])+1
        assert recovered==repeated,'repeat next-use changed recovery'
        resumed_client=MCP(h);after=resumed_client.events(rid,0)['events']
        resumed_client.close()
        assert after[:len(prefix)]==prefix,'committed public prefix changed after engine crash'
        assert [int(event['sequence']) for event in after]==list(range(1,len(after)+1))
        save('r1-before.json',before);save('r1-recovered.json',recovered)
        save('r1-prefix.json',prefix);save('r1-after-events.json',after)
        report['r1_coarse_native_pass']=True
        report['r1_signal_exit']=signal_exit
        report['r1_prefix_events']=len(prefix)
        report['r1_recovered_events']=len(after)
        report['r1_recovered_state']=recovered['state']
        report['r1_repeat_next_use_idempotent']=True
        report['r1_all_internal_background_cuts_qualified']=False
    if args.boundary_matrix:engine_boundary_matrix()
except Exception as error:
    report['error']=str(error);report['traceback']=traceback.format_exc()
finally:
    # Stop only processes started by this probe. No foreign PID or GPU work is touched.
    cleanup=[]
    for home in recovery_homes:
        try:
            stopped=subprocess.run([args.binary,'serve','--stop'],env={**base_env,'JOPEDIME_HOME':str(home)},text=True,capture_output=True,timeout=10)
            commands.append({'argv':['serve','--stop'],'home':str(home),'exit':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr})
        except Exception as error:cleanup.append(str(error))
    for observer in observers:
        try:observer.close()
        except Exception as error:cleanup.append(str(error))
    for child in reversed(children):
        if child.poll() is None:child.terminate()
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        for stream in [child.stdin,child.stdout,child.stderr]:
            if stream is not None:
                try:stream.close()
                except OSError as error:cleanup.append(str(error))
    report['cleanup_errors']=cleanup
    report['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['load_after']=os.getloadavg()[0]
    expected_gates=required_gates(args)
    report['required_gates']=sorted(expected_gates)
    report['missing_gates']=sorted(gate for gate in expected_gates if gate not in report)
    report['failed_gates']=sorted(gate for gate in expected_gates if report.get(gate) is not True)
    report['driver_qualified']=acceptance_passes(report,expected_gates)
    save('commands.json',commands);save('report.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ['traceback','r4_final']}))
if report['driver_qualified'] is not True:
    raise AssertionError('INT-02 acceptance failed, a required mode gate is absent/false, or cleanup failed')
