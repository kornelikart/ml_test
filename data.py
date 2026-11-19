"""Open MOEX ISS snapshot. No credentials. All responses are cached with URLs."""
from pathlib import Path
from datetime import date, timedelta, datetime
import concurrent.futures as cf
import requests, json, gzip, hashlib, time, re, os

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RAW = DATA / 'raw'
BASE = 'https://iss.moex.com/iss/'

def get(path, **params):
    RAW.mkdir(parents=True, exist_ok=True)
    params = dict(params, **{'iss.meta': 'off'})
    url = requests.Request('GET', BASE + path, params=params).prepare().url
    dest = RAW / (hashlib.sha256(url.encode()).hexdigest()[:24] + '.json.gz')
    if dest.exists():
        with gzip.open(dest, 'rt', encoding='utf8') as f: return json.load(f)['response']
    for attempt in range(6):
        try:
            r = requests.get(url, timeout=60); r.raise_for_status(); result = r.json()
            with gzip.open(dest, 'wt', encoding='utf8') as f:
                json.dump({'url':url,'retrieved_utc':datetime.utcnow().isoformat(),'response':result},f,ensure_ascii=False)
            return result
        except Exception:
            if attempt == 5: raise
            time.sleep(1 + attempt * 2)

def table(j, key): return [dict(zip(j[key]['columns'], r)) for r in j[key]['data']]

def history(market, sec=None, **params):
    path=f'history/engines/futures/markets/{market}/securities'
    if sec: path += '/' + sec
    path += '.json'
    out=[]; start=0
    while True:
        j=get(path, start=start, **params); rows=table(j,'history'); out += rows
        cur=j.get('history.cursor',{}).get('data',[[0,len(out),100]])[0]
        if not rows or len(out)>=cur[1]: return out
        start += len(rows)

def spec(sec):
    j=get('securities/'+sec+'.json')
    return {r['name']:r['value'] for r in table(j,'description')}

def candles(job):
    sec, day=job; out=[]; start=0
    while True:
        j=get(f'engines/futures/markets/forts/securities/{sec}/candles.json',
              **{'from':day,'till':day,'interval':1,'start':start})
        rows=table(j,'candles'); out += rows
        if len(rows)<500: break
        start += len(rows)
    # Calendar date only; restrict to common daytime window across session-rule changes.
    out=[r for r in out if r['begin'][:10]==day and '10:00:00'<=r['begin'][11:]<='18:39:00']
    return sec,day,out

def build_snapshot(start='2025-12-01',end='2026-06-30',output='snapshot.json.gz'):
    DATA.mkdir(exist_ok=True)
    mondays=[]; d=date.fromisoformat(start)
    while d.weekday()!=0:d+=timedelta(days=1)
    while d+timedelta(days=4)<=date.fromisoformat(end): mondays.append(d); d+=timedelta(days=7)
    # Selection is made using Monday's published data, entry Tuesday after 10:00.
    symbols=['SiZ5','SiH6','SiM6','SiU6'] if end<='2026-06-30' else ['SiU6','SiZ6','SiH7']
    futures={s:history('forts',s,**{'from':str(date.fromisoformat(start)-timedelta(days=3)),'till':end}) for s in symbols}
    fdict={s:{r['TRADEDATE']:r for r in rs} for s,rs in futures.items()}
    def chain(d): return d, history('options',date=str(d),assetcode='Si')
    with cf.ThreadPoolExecutor(max_workers=6) as ex: chains=dict(ex.map(chain,mondays))
    print('Chains loaded',len(chains),flush=True)
    # The code is only used to group series; expiry and underlying come from official specs.
    representatives={}
    for rows in chains.values():
        for r in rows:
            m=re.fullmatch(r'Si(\d+)(B[A-L]\d[A-Z]?)',r['SECID'])
            if m: representatives.setdefault(m[2],r['SECID'])
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        specs=dict(zip(representatives,ex.map(spec,representatives.values())))
    cycles=[]; missing=[]
    for monday,rows in chains.items():
        rd={r['SECID']:r for r in rows}
        entry=monday+timedelta(days=1); end=monday+timedelta(days=4)
        for target in [7,14,21,28]:
            candidates=[]
            for suffix,sp in specs.items():
                expiry=date.fromisoformat(sp['LSTTRADE']); dte=(expiry-entry).days
                if not 5<=dte<=35 or expiry<=end: continue
                underlying=sp['UNDERLYINGASSET']
                fr=fdict.get(underlying,{}).get(str(monday))
                if not fr or not fr.get('SETTLEPRICE'): continue
                f=float(fr['SETTLEPRICE'])
                avail=[]
                for r in rows:
                    m=re.fullmatch(r'Si(\d+)'+suffix,r['SECID'])
                    if m:
                        k=int(m[1]); put=f'Si{k}B{chr(ord(suffix[1])+12)}{suffix[2:]}'
                        pr=rd.get(put)
                        if pr and (r.get('SETTLEPRICE') or 0)>0 and (pr.get('SETTLEPRICE') or 0)>0:
                            avail.append((abs(k-f),k,r['SECID'],put,r,pr))
                if avail:
                    best=min(avail,key=lambda a:(a[0],a[1]))
                    candidates.append((abs(dte-target),dte,suffix,sp,best,f))
            if not candidates:
                missing.append({'selection':str(monday),'target_dte':target,'reason':'No valid paired series / Monday settlement'})
                continue
            _,dte,suffix,sp,best,f=min(candidates,key=lambda a:(a[0],a[1],a[2]))
            _,k,call,put,cr,pr=best
            cycles.append(dict(id=str(entry),selection=str(monday),entry=str(entry),end=str(end),
                target_dte=target,dte=dte,expiry=sp['LSTTRADE'],underlying=sp['UNDERLYINGASSET'],
                strike=k,call=call,put=put,selection_future=f,
                selection_premium=float(cr['SETTLEPRICE']+pr['SETTLEPRICE']),
                selection_call_volume=cr.get('VOLUME') or 0,selection_put_volume=pr.get('VOLUME') or 0))
    print('Selected cycles',len(cycles),'missing',len(missing),flush=True)
    jobs=sorted({(c['underlying'],str(date.fromisoformat(c['entry'])+timedelta(days=i))) for c in cycles for i in range(4)})
    bars={}
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for idx,(sec,day,rows) in enumerate(ex.map(candles,jobs)):
            bars.setdefault(sec,{})[day]=rows
            if idx%25==0: print('Minute dates',idx+1,'/',len(jobs),flush=True)
    # Strike is chosen at Tuesday open from Monday's already listed pairs.
    # This preserves ATM selection after an overnight move without using Tuesday option quotes.
    for c in cycles:
        minute=bars[c['underlying']].get(c['entry'],[])
        if not minute: continue
        opening=minute[0]['open']; suffix=re.fullmatch(r'Si\d+(B[A-L]\d[A-Z]?)',c['call'])[1]
        chain_rows=chains[date.fromisoformat(c['selection'])]; rd={r['SECID']:r for r in chain_rows}; avail=[]
        for r in chain_rows:
            m=re.fullmatch(r'Si(\d+)'+suffix,r['SECID'])
            if not m:continue
            k=int(m[1]);put=f'Si{k}B{chr(ord(suffix[1])+12)}{suffix[2:]}'
            pr=rd.get(put)
            if pr and (r.get('SETTLEPRICE') or 0)>0 and (pr.get('SETTLEPRICE') or 0)>0:
                avail.append((abs(k-opening),k,r,pr))
        if avail:
            _,k,cr,pr=min(avail,key=lambda a:(a[0],a[1]))
            c.update(strike=k,call=cr['SECID'],put=pr['SECID'],selection_premium=float(cr['SETTLEPRICE']+pr['SETTLEPRICE']),
                     selection_call_volume=cr.get('VOLUME') or 0,selection_put_volume=pr.get('VOLUME') or 0)
    optjobs=sorted({(c[x],c['selection'],c['end']) for c in cycles for x in ['call','put']})
    def oh(j): return '|'.join(j),history('options',j[0],**{'from':j[1],'till':j[2]})
    with cf.ThreadPoolExecutor(max_workers=6) as ex: options=dict(ex.map(oh,optjobs))
    snapshot=dict(cycles=cycles,missing=missing,futures=futures,options=options,bars=bars,
                  series_specs=specs,created_utc=datetime.utcnow().isoformat())
    with gzip.open(DATA/output,'wt',encoding='utf8') as f: json.dump(snapshot,f,ensure_ascii=False)
    print('Snapshot saved; minute rows',sum(len(rs) for ds in bars.values() for rs in ds.values()),flush=True)
    return snapshot

def load_snapshot():
    with gzip.open(DATA/os.environ.get('FAMILYBROKER_SNAPSHOT','snapshot.json.gz'),'rt',encoding='utf8') as f:return json.load(f)

if __name__=='__main__': build_snapshot()
