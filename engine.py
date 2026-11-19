"""Minute research simulation. Option values are model marks, not executable quotes."""
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timedelta
from functools import lru_cache
import math
import numpy as np
from scipy.special import ndtr
from data import load_snapshot

YEAR=365.0*86400
STRATEGIES=['combined','straddle','grid','delta_hedged']

@dataclass(frozen=True)
class Config:
    target_dte:int=14
    step_fraction:float=0.125
    gamma_scale:float=1.0
    fixed_lots:int=0  # 0 = round N*Gamma*step, at least one whole contract
    width:float=1.0
    n_options:int=20
    inventory_cap:int=20
    hedge_threshold:float=2.0
    capital:float=1_000_000
    maker_fee:float=2.0
    taker_fee:float=4.0
    option_fee:float=5.0
    option_half_spread:float=0.025
    option_spread_floor:float=10.0
    slippage:float=2.0
    participation:float=0.01
    margin_future:float=0.15
    margin_option:float=0.05 # per long leg, assumed gross; no offsets
    capital_rate:float=0.15
    min_option_volume:int=0  # prior selection-day volume on each leg; optional liquidity screen
    strict_execution:bool=True
    enforce_capital:bool=True
    adaptive_gamma:bool=False
    boundary_exit:bool=False
    valuation_clock:str='23:59:59' # explicit assumption, NOT an observed settlement timestamp
    start:str='2025-12-01'
    end:str='2026-06-30'

def black(F,K,t,v):
    F=np.asarray(F,dtype=float); t=np.maximum(t,1e-10); v=np.maximum(v,1e-6)
    w=v*np.sqrt(t); d1=np.log(F/K)/w+w/2; d2=d1-w
    pdf=np.exp(-d1*d1/2)/np.sqrt(2*np.pi)
    call=F*ndtr(d1)-K*ndtr(d2); put=call-F+K
    return call+put,2*ndtr(d1)-1,2*pdf/(F*w),-F*pdf*v/np.sqrt(t),2*F*pdf*np.sqrt(t)

def implied(F,K,t,P):
    if not np.isfinite(P) or P<abs(F-K) or t<=0 or F<=0 or K<=0: return None
    lo,hi=0.000001,3.0
    if P>float(black(F,K,t,hi)[0]): return None
    for _ in range(55):
        mid=(lo+hi)/2
        if black(F,K,t,mid)[0]>P: hi=mid
        else: lo=mid
    return (lo+hi)/2

@lru_cache(maxsize=1)
def snapshot(): return load_snapshot()

@lru_cache(maxsize=160)
def prepare(cycle_id,target,valuation_clock='23:59:59'):
    return prepare_snapshot(snapshot(),cycle_id,target,valuation_clock)

def prepare_snapshot(snap,cycle_id,target,valuation_clock='23:59:59'):
    c=next(c for c in snap['cycles'] if c['id']==cycle_id and c['target_dte']==target)
    s=c['underlying']; expiry=datetime.fromisoformat(c['expiry']+' 18:45:00')
    fr={r['TRADEDATE']:r for r in snap['futures'][s]}
    legs=[]
    for name in ['call','put']:
        key='|'.join([c[name],c['selection'],c['end']])
        legs.append({r['TRADEDATE']:r for r in snap['options'][key]})
    ivmarks=[]
    for day in sorted(set(legs[0])&set(legs[1])&set(fr)):
        cp=legs[0][day].get('SETTLEPRICE'); pp=legs[1][day].get('SETTLEPRICE'); F=fr[day].get('SETTLEPRICE')
        if cp is None or pp is None or F is None: continue
        # Settlement observation convention: end of calendar day, usable only next date.
        ts=datetime.fromisoformat(day+' '+valuation_clock); t=(expiry-ts).total_seconds()/YEAR
        iv=implied(F,c['strike'],t,cp+pp)
        if iv is not None: ivmarks.append((day,iv,cp+pp,F))
    rows=[]
    for day,rs in sorted(snap['bars'].get(s,{}).items()):
        if c['entry']<=day<=c['end']: rows+=rs
    if not rows or len(rows)<400: return None,'insufficient minute bars'
    # Never silently move Tuesday entry to another day.
    if rows[0]['begin'][:10]!=c['entry'] or rows[-1]['begin'][:10]!=c['end']:return None,'entry or exit day missing'
    dates=[r['begin'][:10] for r in rows]
    vols=[]; stale=[]
    for day in dates:
        known=[x for x in ivmarks if x[0]<day]
        if not known:return None,'no prior-day valid option IV'
        last=known[-1]; vols.append(last[1]); stale.append((datetime.fromisoformat(day)-datetime.fromisoformat(last[0])).days)
    # First observation is entry at first bar open; others are close marks at end of minute.
    ts=[datetime.fromisoformat(rows[0]['begin'])]+[datetime.fromisoformat(r['end']) for r in rows]
    F=np.array([rows[0]['open']]+[r['close'] for r in rows],float)
    v=np.array([vols[0]]+vols); tau=np.array([(expiry-x).total_seconds()/YEAR for x in ts])
    if tau.min()<=0:return None,'expiry reached'
    p,d,g,theta,vega=black(F,c['strike'],tau,v)
    df=np.diff(F); dt=np.array([(b-a).total_seconds()/YEAR for a,b in zip(ts,ts[1:])]); dv=np.diff(v)
    parts=np.stack([d[:-1]*df,0.5*g[:-1]*df**2,theta[:-1]*dt,vega[:-1]*dv],axis=1)
    residual=np.diff(p)-parts.sum(axis=1)
    elapsed=(ts[-1]-ts[0]).total_seconds()/YEAR
    # Five-minute sampled close-to-close quadratic variation, including overnight gaps.
    sample=[F[0]]; last_bucket=None
    for r in rows:
        bucket=r['begin'][:14]+str(int(r['begin'][14:16])//5)
        if bucket!=last_bucket: sample.append(r['close']); last_bucket=bucket
        else: sample[-1]=r['close']
    if sample[-1]!=F[-1]: sample.append(F[-1])
    lr=np.diff(np.log(sample)); rv=math.sqrt(float(lr@lr)/elapsed)
    trend=abs(math.log(F[-1]/F[0]))/max(math.sqrt(float(lr@lr)),1e-9)
    return dict(c=c,rows=rows,ts=ts,F=F,iv=v,p=p,delta=d,gamma=g,theta=theta,vega=vega,
                parts=parts,residual=residual,rv=rv,trend_score=trend,elapsed=elapsed,
                stale_max=max(stale),ivmarks=ivmarks),None

def close_at_next_open(pre, index):
    """Prefix ending at the NEXT observed open; retains overnight gap and updated known IV."""
    short=dict(pre); short['rows']=[dict(r) for r in pre['rows'][:index+1]]
    last=short['rows'][-1]; op=last['open']
    last.update(high=op,low=op,close=op,end=last['begin'],volume=0)
    short['ts']=pre['ts'][:index+2].copy();short['ts'][-1]=datetime.fromisoformat(last['begin'])
    short['F']=pre['F'][:index+2].copy();short['F'][-1]=op
    short['iv']=pre['iv'][:index+2].copy()
    expiry=datetime.fromisoformat(pre['c']['expiry']+' 18:45:00')
    tau=np.array([(expiry-t).total_seconds()/YEAR for t in short['ts']])
    p,d,g,th,ve=black(short['F'],pre['c']['strike'],tau,short['iv'])
    short.update(p=p,delta=d,gamma=g,theta=th,vega=ve,forced_exit=True)
    df=np.diff(short['F']);dt=np.array([(b-a).total_seconds()/YEAR for a,b in zip(short['ts'],short['ts'][1:])])
    short['parts']=np.stack([d[:-1]*df,.5*g[:-1]*df**2,th[:-1]*dt,ve[:-1]*np.diff(short['iv'])],axis=1)
    short['residual']=np.diff(p)-short['parts'].sum(axis=1)
    sample=[short['F'][0]];bucket_previous=None
    for row in short['rows']:
        bucket=row['begin'][:14]+str(int(row['begin'][14:16])//5)
        if bucket!=bucket_previous:sample.append(row['close']);bucket_previous=bucket
        else:sample[-1]=row['close']
    elapsed=max((short['ts'][-1]-short['ts'][0]).total_seconds()/YEAR,1e-12)
    lr=np.diff(np.log(sample));rv=math.sqrt(float(lr@lr)/elapsed)
    short.update(rv=rv,elapsed=elapsed,trend_score=abs(math.log(short['F'][-1]/short['F'][0]))/max(math.sqrt(float(lr@lr)),1e-9))
    return short

def simulate(pre,cfg,strategy,detail=False,starting_equity=None):
    starting_equity=cfg.capital if starting_equity is None else starting_equity
    c=pre['c']; rows=pre['rows']; F=pre['F']; p=pre['p']; N=cfg.n_options
    hasopt=strategy!='grid'; grid=strategy in ('grid','combined'); hedge=strategy=='delta_hedged'
    h=max(1,round(p[0]*cfg.step_fraction)); low=c['strike']-cfg.width*p[0]; high=c['strike']+cfg.width*p[0]
    q=cfg.fixed_lots or max(1,round(N*pre['gamma'][0]*h*cfg.gamma_scale))
    anchor=c['strike']+round((F[0]-c['strike'])/h)*h
    inventory=0; gross=0.; fees=0.; spread=0.; slip=0.; capital_cost=0.; funding=0.
    option_fees=4*N*cfg.option_fee if hasopt else 0.
    # Half-spread applied to the straddle; floor is per option leg.
    option_spread=N*(max(p[0]*cfg.option_half_spread,2*cfg.option_spread_floor)+
                     max(p[-1]*cfg.option_half_spread,2*cfg.option_spread_floor)) if hasopt else 0.
    option_entry_cost=(2*N*cfg.option_fee+N*max(p[0]*cfg.option_half_spread,2*cfg.option_spread_floor)) if hasopt else 0.
    active=bool(low<F[0]<high); pending=0; fills=[]; path=[]; ambiguous=0; limited=0
    peak_margin=0.; peak_inv=0; max_total_delta=abs(N*pre['delta'][0]) if hasopt else 0.
    breach=None; max_liquidity=0.; min_free=float('inf');blocked=0;boundary_pending=False
    entry_margin=(2*N*cfg.margin_option*F[0]) if hasopt else 0.
    peak_margin=entry_margin;max_liquidity=entry_margin+option_entry_cost
    min_free=starting_equity-entry_margin-2*option_entry_cost
    path.append(dict(time=pre['ts'][0].isoformat(' '),price=F[0],inventory=0,equity=starting_equity-option_entry_cost,
                     delta=N*pre['delta'][0] if hasopt else 0,margin=entry_margin,option_pnl=0,grid_pnl=0,costs=option_entry_cost))
    for i,r in enumerate(rows,1):
        old=inventory; fill_qty=0; fill_price=0.; execution=''
        new_session=i==1 or r['begin'][:10]!=rows[i-2]['begin'][:10]
        forced_last=pre.get('forced_exit',False) and i==len(rows)
        if cfg.strict_execution and new_session:pending=0
        if cfg.adaptive_gamma:
            q=cfg.fixed_lots or max(1,round(N*pre['gamma'][i-1]*h*cfg.gamma_scale))
        if boundary_pending and inventory and not forced_last:
            fill_qty=-inventory;fill_price=r['open'];execution='taker';boundary_pending=False
        if hedge and pending and not forced_last:
            # Decision uses previous close delta, execution is at following bar open.
            qty=int(np.clip(pending,-cfg.inventory_cap-inventory,cfg.inventory_cap-inventory))
            if qty:
                fill_qty=qty; fill_price=r['open']; execution='taker'
            pending=0
        if grid and active and not forced_last:
            if not low<r['open']<high: active=False
            else:
                buy=anchor-h; sell=anchor+h
                b=(low<=buy<=high and r['low']<=buy-1 and inventory+q<=cfg.inventory_cap)
                s=(low<=sell<=high and r['high']>=sell+1 and inventory-q>=-cfg.inventory_cap)
                if cfg.strict_execution:
                    # Cancel overnight; first minute observes the market. Each subsequent
                    # cancel/replace uses the PREVIOUS close, with post-only eligibility.
                    if new_session:
                        blocked+=int(b or s);b=s=False
                    else:
                        blocked+=int(b and buy>=F[i-1])+int(s and sell<=F[i-1])
                        b=b and buy<F[i-1];s=s and sell>F[i-1]
                if b and s: ambiguous+=1
                elif b or s:
                    if math.floor(r['volume']*cfg.participation)>=q:
                        fill_qty=q if b else -q; fill_price=buy if b else sell; execution='maker'; anchor=fill_price
                    else: limited+=1
            if r['low']<=low or r['high']>=high: active=False
        # Exact futures ledger: old inventory marks prev close -> this close,
        # plus this bar's transaction marked from fill price to this close.
        gross += old*(F[i]-F[i-1])+fill_qty*(F[i]-fill_price)
        if fill_qty:
            inventory+=fill_qty
            fee=abs(fill_qty)*(cfg.maker_fee if execution=='maker' else cfg.taker_fee)
            slippage=abs(fill_qty)*cfg.slippage if execution=='taker' else 0.
            fees+=fee; slip+=slippage
            trade_time = r['begin'] if execution == 'taker' else r['end']
            fills.append(dict(time=trade_time,qty=fill_qty,price=fill_price,kind=execution,
                              inventory=inventory,total_delta=(N*pre['delta'][i] if hasopt else 0)+inventory,
                              fee=fee,slippage=slippage))
        elapsed=(pre['ts'][i]-pre['ts'][i-1]).total_seconds()/YEAR
        # Charge previous holding's gross margin, including overnight time, no collateral yield.
        margin_prev=abs(old)*cfg.margin_future*F[i-1]+(2*N*cfg.margin_option*F[i-1] if hasopt else 0)
        capital_cost+=margin_prev*cfg.capital_rate*elapsed
        margin=abs(inventory)*cfg.margin_future*F[i]+(2*N*cfg.margin_option*F[i] if hasopt else 0)
        peak_margin=max(peak_margin,margin); peak_inv=max(peak_inv,abs(inventory))
        optpnl=N*(p[i]-p[0]) if hasopt else 0.
        net=optpnl+gross-fees-slip-capital_cost-option_entry_cost
        totaldelta=(N*pre['delta'][i] if hasopt else 0)+inventory
        max_total_delta=max(max_total_delta,abs(totaldelta))
        max_liquidity=max(max_liquidity,margin+max(0,-net))
        exit_reserve=abs(inventory)*(cfg.taker_fee+cfg.slippage)+(2*N*cfg.option_fee+N*max(p[i]*cfg.option_half_spread,2*cfg.option_spread_floor) if hasopt else 0)
        free=starting_equity+net-margin-exit_reserve
        min_free=min(min_free,free)
        if cfg.enforce_capital and free<0 and i<len(rows):
            closed=simulate(close_at_next_open(pre,i),replace(cfg,enforce_capital=False),strategy,detail,starting_equity)
            closed.update(forced_liquidation=True,capital_breach=True,risk_trigger=pre['ts'][i].isoformat(' '))
            return closed
        if grid and breach is None and not active:
            breach=dict(time=pre['ts'][i].isoformat(' '),price=F[i],inventory=inventory,
                        futures_pnl=gross,option_pnl=optpnl,net_pnl=net,total_delta=totaldelta)
            boundary_pending=cfg.boundary_exit
        if hedge and abs(totaldelta)>=cfg.hedge_threshold: pending=-round(totaldelta)
        if detail or i==len(rows):
            path.append(dict(time=pre['ts'][i].isoformat(' '),price=F[i],inventory=inventory,
                             equity=starting_equity+net,delta=totaldelta,margin=margin,option_pnl=optpnl,
                             grid_pnl=gross,costs=fees+slip+capital_cost+option_entry_cost))
    # Close any futures at last observed close, with taker costs; options close at lag-IV model mark.
    if inventory:
        fees+=abs(inventory)*cfg.taker_fee; slip+=abs(inventory)*cfg.slippage
        fills.append(dict(time=pre['ts'][-1].isoformat(' '),qty=-inventory,price=F[-1],kind='exit',inventory=0,
                          total_delta=0,fee=abs(inventory)*cfg.taker_fee,slippage=abs(inventory)*cfg.slippage))
    optgross=N*(p[-1]-p[0]) if hasopt else 0.
    parts=pre['parts'].sum(axis=0)*N if hasopt else np.zeros(4)
    residual=pre['residual'].sum()*N if hasopt else 0.
    net=optgross+gross-fees-option_fees-option_spread-slip-capital_cost-funding
    path[-1].update(equity=starting_equity+net,inventory=0,delta=0,
                    costs=fees+option_fees+option_spread+slip+capital_cost,margin=0)
    result=dict(cycle=c['id'],strategy=strategy,target_dte=cfg.target_dte,dte=c['dte'],underlying=c['underlying'],
                call=c['call'],put=c['put'],strike=c['strike'],start_price=F[0],end_price=F[-1],
                iv_entry=pre['iv'][0],rv=pre['rv'],rv_iv=pre['rv']/pre['iv'][0],trend_score=pre['trend_score'],
                option_pnl=optgross,delta_pnl=parts[0],gamma_pnl=parts[1],theta_pnl=parts[2],vega_pnl=parts[3],
                residual_pnl=residual,futures_pnl=gross,futures_fees=fees,option_fees=option_fees,
                option_spread=option_spread,slippage=slip,capital_cost=capital_cost,funding=funding,pnl=net,
                step=h,lots=(cfg.fixed_lots or max(1,round(N*pre['gamma'][0]*h*cfg.gamma_scale))),lower=low,upper=high,fills=len(fills),peak_inventory=peak_inv,peak_margin=peak_margin,
                max_liquidity_need=max_liquidity,capital_breach=max_liquidity>cfg.capital,
                starting_equity=starting_equity,ending_equity=starting_equity+net,
                min_free_equity=min_free,forced_liquidation=False,risk_trigger=None,blocked_post_only=blocked,
                max_total_delta=max_total_delta,ambiguous_bars=ambiguous,volume_blocked=limited,
                boundary_breached=breach is not None,stale_iv_days=pre['stale_max'],bars=len(rows),
                call_volume=c['selection_call_volume'],put_volume=c['selection_put_volume'])
    result['capital_breach']=min_free<0
    if detail:result.update(path=path,transactions=fills,breach=breach)
    return result

def run(cfg=Config(),detail=False,strategies=STRATEGIES):
    results=[]; skipped=[]
    equity={s:cfg.capital for s in strategies}
    for c in snapshot()['cycles']:
        if c['target_dte']!=cfg.target_dte or not cfg.start<=c['entry']<=cfg.end or c['end']>cfg.end:continue
        if min(c['selection_call_volume'],c['selection_put_volume'])<cfg.min_option_volume:
            skipped.append({'cycle':c['id'],'reason':'prior option volume below configured minimum'});continue
        pre,error=prepare(c['id'],cfg.target_dte,cfg.valuation_clock)
        if error: skipped.append({'cycle':c['id'],'reason':error});continue
        for strategy in strategies:
            required=(2*cfg.n_options*cfg.margin_option*pre['F'][0]+4*cfg.n_options*cfg.option_fee+
                      2*cfg.n_options*max(pre['p'][0]*cfg.option_half_spread,2*cfg.option_spread_floor)) if strategy!='grid' else 0
            if cfg.enforce_capital and equity[strategy]<=required:
                skipped.append({'cycle':c['id'],'strategy':strategy,'reason':'insufficient account equity at entry'});continue
            result=simulate(pre,cfg,strategy,detail,equity[strategy])
            results.append(result);equity[strategy]+=result['pnl']
    return results,skipped

def sensitivity(base=Config()):
    # One factor at a time; no in-sample optimization or selection of a winning specification.
    variants=[('base','base',base)]
    for key,values in [('step_fraction',[.0625,.25]),('fixed_lots',[1,2,3]),('width',[.5,1.5]),('target_dte',[7,21,28]),
                       ('option_half_spread',[.0,.01,.05]),('slippage',[5.0]),('min_option_volume',[20])]:
        for value in values:variants.append((key,str(value),replace(base,**{key:value})))
    out=[]
    for key,val,cfg in variants:
        rs,sk=run(cfg,strategies=['combined'])
        pn=np.array([r['pnl'] for r in rs])
        out.append(dict(parameter=key,value=val,cycles=len(rs),pnl=float(pn.sum()),mean=float(pn.mean()) if len(pn) else 0,
                        worst=float(pn.min()) if len(pn) else 0,mean_lots=float(np.mean([r['lots'] for r in rs])),
                        mean_dte=float(np.mean([r['dte'] for r in rs])),skipped=len(sk)))
    return out
