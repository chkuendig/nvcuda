"""Validate host-only diagnostic fixtures and exact payload bytes."""
import hashlib,json,os,pathlib,struct,subprocess,sys,tempfile
binary=pathlib.Path(sys.argv[1]).resolve()
root=pathlib.Path(sys.argv[2]).resolve();root.mkdir(mode=0o700)
for case in ('disabled','pixels','readback','last-write-failure'):
    cap=root/case;cap.mkdir(mode=0o700)
    env={k:v for k,v in os.environ.items() if k not in ('ZLUDA_D3D11_COPY_PROBE_DIR','ZLUDA_D3D11_COPY_PROBE_READBACK')}
    if case!='disabled':env['ZLUDA_D3D11_COPY_PROBE_DIR']=str(cap)
    if case in ('disabled','readback'):env['ZLUDA_D3D11_COPY_PROBE_READBACK']='1'
    run=subprocess.run([str(binary),case],env=env,capture_output=True,text=True,timeout=10)
    (root/(case+'.stdout')).write_text(run.stdout);(root/(case+'.stderr')).write_text(run.stderr)
    if run.returncode or 'PASS ' not in run.stdout:raise RuntimeError((case,run.returncode,run.stdout,run.stderr))
    if case=='disabled':
        if list(cap.iterdir()):raise RuntimeError('disabled output')
        continue
    summaries=list(cap.glob('*-summary.json'))
    if len(summaries)!=1:raise RuntimeError('missing/ambiguous summary')
    summary=json.loads(summaries[0].read_text())
    if not summary['complete']:raise RuntimeError('incomplete summary')
    if case=='last-write-failure':
        if summary['captured']!=2 or summary['written']!=1 or summary['io_errors']!=1:raise RuntimeError('final output error lost')
    else:
        records=[json.loads(p.read_text()) for p in sorted(cap.glob('*-0*.json'))]
        if len(records)!=2:raise RuntimeError('record count')
        good,bad=records
        if good['cuda_result'] or bad['cuda_result']!=999:raise RuntimeError('API result')
        readback=good['readback']
        if case=='readback':
            if readback!=dict(attempted=1,hresult=0,row_pitch=48,fnv1a64=good['fnv1a64'],mismatched_bytes=0,start_ns=0,end_ns=0,first_mismatch=None):raise RuntimeError('readback comparison')
        elif readback['attempted']:raise RuntimeError('unexpected readback')
        if bad['readback']['attempted']:raise RuntimeError('failed-copy readback')
        if good['channels']!=[
            dict(finite=4,nonzero=2,nan=0,inf=0,min=-2,max=3),
            dict(finite=4,nonzero=3,nan=0,inf=0,min=-4,max=2),
            dict(finite=2,nonzero=2,nan=1,inf=1,min=5,max=7),
            dict(finite=3,nonzero=3,nan=0,inf=1,min=4,max=8)]:raise RuntimeError('channel stats')
        if any(c['min'] is not None or c['max'] is not None for c in bad['channels']):raise RuntimeError('invalid ranges')
        payload=(cap/good['payload']).read_bytes()
        if len(payload)!=64 or struct.unpack('<4f',payload[:16])[:2]!=(0,1):raise RuntimeError('payload')
        value=14695981039346656037
        for b in payload:value=((value^b)*1099511628211)&((1<<64)-1)
        if good['fnv1a64']!=f'{value:016x}':raise RuntimeError('hash')
        if summary['dropped']!=1 or summary['io_errors'] or summary['written']!=2:raise RuntimeError('terminal counters')
    if list(cap.glob('*.pending')):raise RuntimeError('unfinished publication')
    print('PASS',case)
print('PASS all copy-output host cases')
