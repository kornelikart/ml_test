"""Reproduce tables and figures from the archived snapshot without network access."""
from pathlib import Path
from dataclasses import asdict
import json, gzip, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import Config, run, sensitivity, snapshot, prepare, black

ROOT=Path(__file__).resolve().parent
NAMES={'combined':'Straddle + grid','straddle':'Straddle only','grid':'Grid only','delta_hedged':'Delta hedge'}
STYLES={'combined':('-','black'),'straddle':('--','0.35'),'grid':(':','0.5'),'delta_hedged':('-.','0.15')}
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x
def write_json(path,x):path.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf8')

def calculate(cfg=Config(),out=None,with_sensitivity=True):
    out=Path(out or ROOT/'results');out.mkdir(exist_ok=True,parents=True)
    rs,skipped=run(cfg,detail=True)
    if not rs:raise ValueError('Нет полных циклов в выбранном периоде')
    df=pd.DataFrame([{k:v for k,v in r.items() if k not in ['path','transactions','breach']} for r in rs])
    df['regime']=np.where(df.rv_iv>=1.5,'volatility spike',np.where(df.trend_score>=.8,'trend','range'))
    df.to_csv(out/'cycles.csv',index=False,encoding='utf-8-sig')
    write_json(out/'config.json',asdict(cfg));write_json(out/'excluded.json',dict(data_selection=snapshot()['missing'],simulation=skipped))
    paths=[]; trades=[]; breaches=[]; metrics=[]; eqs={}
    for strategy in NAMES:
        pieces=[]
        for r in [x for x in rs if x['strategy']==strategy]:
            path=pd.DataFrame(r['path'])
            path['strategy']=strategy;path['cycle']=r['cycle'];pieces.append(path);paths.append(path)
            trades.extend([dict(strategy=strategy,cycle=r['cycle'],**t) for t in r['transactions']])
            if r['breach']:breaches.append(dict(strategy=strategy,cycle=r['cycle'],**r['breach']))
        eq=pd.concat(pieces,ignore_index=True);eq['time']=pd.to_datetime(eq['time']);eq=eq.set_index('time')
        eqs[strategy]=eq
        days=pd.bdate_range(cfg.start,cfg.end)
        daily=eq.equity.resample('D').last().reindex(days).ffill().fillna(cfg.capital)
        ret=daily.pct_change().fillna((daily.iloc[0]-cfg.capital)/cfg.capital)
        series=np.r_[cfg.capital,eq.equity.to_numpy()]; peak=np.maximum.accumulate(series)
        sub=df[df.strategy==strategy];pn=sub.pnl.sort_values();count=max(1,math.ceil(.05*len(pn)))
        metrics.append(dict(strategy=strategy,cycles=len(pn),pnl=float(pn.sum()),return_pct=float(pn.sum()/cfg.capital*100),
             sharpe=float(ret.mean()/ret.std(ddof=1)*np.sqrt(252)) if ret.std() else None,
             max_drawdown_rub=float((peak-series).max()),max_drawdown_pct=float(((peak-series)/peak).max()*100),
             win_rate=float((pn>0).mean()),worst=float(pn.min()),best=float(pn.max()),
             var_5=float(pn.quantile(.05)),es_worst_5=float(pn.iloc[:count].mean()),tail_count=count,
             peak_margin=float(sub.peak_margin.max()),max_liquidity_need=float(sub.max_liquidity_need.max()),
             capital_breaches=int(sub.capital_breach.sum()),forced_liquidations=int(sub.forced_liquidation.sum()),
             min_free_equity=float(sub.min_free_equity.min())))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.concat(paths).to_csv(out/'minute_ledger.csv.gz',index=False,compression='gzip')
    pd.DataFrame(trades).to_csv(out/'transactions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(breaches).to_csv(out/'boundary_events.csv',index=False,encoding='utf-8-sig')
    components=['delta_pnl','gamma_pnl','theta_pnl','vega_pnl','residual_pnl','futures_pnl',
                'futures_fees','option_fees','option_spread','slippage','capital_cost','funding','pnl']
    decomp=df.groupby('strategy')[components].sum();decomp.to_csv(out/'decomposition.csv',encoding='utf-8-sig')
    regimes=df.groupby(['strategy','regime']).agg(cycles=('pnl','count'),pnl=('pnl','sum'),mean=('pnl','mean')).reset_index()
    regimes.to_csv(out/'regimes.csv',index=False,encoding='utf-8-sig')
    base=df[df.strategy=='combined'].sort_values('cycle'); pn=base.pnl
    tails=base.nsmallest(max(1,math.ceil(.05*len(base))),'pnl');tails.to_csv(out/'worst_5_percent.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':150,'figure.constrained_layout.use':True})
    fig,ax=plt.subplots(figsize=(10,4))
    for s,eq in eqs.items():
        ls,color=STYLES[s];ax.plot(eq.index,eq.equity/1000,ls=ls,color=color,label=NAMES[s],linewidth=1.3)
    ax.axhline(cfg.capital/1000,color='0.7',lw=.6);ax.set_ylabel('Equity, RUB thousand');ax.legend(ncol=2);ax.grid(alpha=.15)
    fig.savefig(out/'equity.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    axs[0].hist(pn/1000,bins=10,color='0.75',edgecolor='black');axs[0].axvline(0,color='black',ls='--');axs[0].set(xlabel='Cycle P&L, RUB thousand',ylabel='Cycles')
    for regime,sub in base.groupby('regime'):
        marker={'trend':'^','range':'o','volatility spike':'s'}[regime]
        axs[1].scatter(sub.rv_iv,sub.pnl/1000,marker=marker,facecolors='none',edgecolors='black',label=regime)
    axs[1].axhline(0,color='0.5',lw=.7);axs[1].axvline(1,color='0.5',ls=':');axs[1].set(xlabel='Realized / entry implied volatility',ylabel='Cycle P&L, RUB thousand');axs[1].legend(fontsize=8)
    fig.savefig(out/'distribution_rv.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    labels=['Delta','Gamma','Theta','Vega','Residual','Grid','Fees','Spread','Slippage','Capital','Funding']
    dc=decomp.loc['combined']; vals=[dc.delta_pnl,dc.gamma_pnl,dc.theta_pnl,dc.vega_pnl,dc.residual_pnl,dc.futures_pnl,
        -dc.futures_fees-dc.option_fees,-dc.option_spread,-dc.slippage,-dc.capital_cost,-dc.funding]
    ax.bar(labels,np.array(vals)/1000,color=['0.35' if v>=0 else '0.8' for v in vals],edgecolor='black',linewidth=.6)
    ax.tick_params(axis='x',rotation=35);ax.set_ylabel('P&L, RUB thousand');ax.axhline(0,color='black',lw=.5)
    fig.savefig(out/'decomposition.png');plt.close(fig)
    selected={'best':base.loc[pn.idxmax(),'cycle'],'worst':base.loc[pn.idxmin(),'cycle'],
              'typical':base.loc[(pn-pn.median()).abs().idxmin(),'cycle']}
    examples={}
    for label,cid in selected.items():
        r=next(x for x in rs if x['strategy']=='combined' and x['cycle']==cid)
        path=pd.DataFrame(r['path']);path.time=pd.to_datetime(path.time)
        fig,axes=plt.subplots(3,1,figsize=(10,6.5),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
        axes[0].plot(path.time,path.price,color='black',lw=.8)
        for level in np.arange(r['strike'],r['lower']-r['step'],-r['step']):
            if level>=r['lower']:axes[0].axhline(level,color='0.75',lw=.45)
        for level in np.arange(r['strike']+r['step'],r['upper']+r['step'],r['step']):
            if level<=r['upper']:axes[0].axhline(level,color='0.75',lw=.45)
        for level in [r['lower'],r['upper']]:axes[0].axhline(level,color='black',ls='--',lw=1)
        tr=pd.DataFrame(r['transactions'])
        if len(tr):
            for sign,marker in [(1,'^'),(-1,'v')]:
                ss=tr[np.sign(tr.qty)==sign];axes[0].scatter(pd.to_datetime(ss.time),ss.price,marker=marker,s=18,color='0.25')
        axes[0].set(title=f'{label.title()} | {cid} | P&L {r["pnl"]:,.0f} RUB',ylabel='Si price, RUB')
        axes[1].step(path.time,path.inventory,where='post',color='black');axes[1].set_ylabel('Futures lots')
        axes[2].plot(path.time,path.delta,color='black',lw=.8);axes[2].axhline(0,color='0.6',lw=.5);axes[2].set_ylabel('Total delta')
        fig.savefig(out/f'cycle_{label}.png');plt.close(fig)
        examples[label]={k:v for k,v in r.items() if k not in ['path','transactions']}
    # Stress the breach cycle with the largest terminal loss on the grid leg.
    breached=[x for x in rs if x['strategy']=='combined' and x['breach']]
    worst=min(breached,key=lambda x:x['futures_pnl']) if breached else None
    stress=[]
    if worst:
        pre,_=prepare(worst['cycle'],cfg.target_dte);br=worst['breach'];F0=br['price'];K=worst['strike'];inv=br['inventory']
        idx=next(i for i,t in enumerate(pre['ts']) if t.isoformat(' ')==br['time'])
        expiry=__import__('datetime').datetime.fromisoformat(pre['c']['expiry']+' 18:45:00')
        tau=(expiry-pre['ts'][idx]).total_seconds()/(365*86400)
        for shock in [-.2,-.1,.1,.2]:
            Fs=F0*(1+shock);p0=black(F0,K,tau,pre['iv'][idx])[0];p1=black(Fs,K,tau,pre['iv'][idx])[0]
            for multiple in [1,2,3]:
                ip=inv*multiple;op=cfg.n_options*(p1-p0);fp=ip*(Fs-F0)
                stress.append(dict(cycle=worst['cycle'],breach_price=F0,shock=shock,inventory_multiple=multiple,inventory=ip,option_change=float(op),
                                   futures_change=float(fp),combined_change=float(op+fp),wing_lots=cfg.n_options))
    pd.DataFrame(stress).to_csv(out/'tail_stress.csv',index=False,encoding='utf-8-sig')
    sens=sensitivity(cfg) if with_sensitivity else []
    if sens:pd.DataFrame(sens).to_csv(out/'sensitivity.csv',index=False,encoding='utf-8-sig')
    summary=dict(metrics=metrics,decomposition=decomp.reset_index().to_dict('records'),examples=examples,
                 sensitivity=sens,regimes=regimes.to_dict('records'),skipped=skipped,
                 first_cycle=str(base.cycle.min()),last_cycle=str(base.cycle.max()),
                 mean_iv=float(base.iv_entry.mean()),mean_rv=float(base.rv.mean()),
                 rv_iv_correlation=float(base.rv_iv.corr(base.pnl)),
                 rv_high_count=int((base.rv_iv>1).sum()),rv_high_profit_count=int(((base.rv_iv>1)&(base.pnl>0)).sum()),
                 low_volume_cycles=int(((base.call_volume<cfg.n_options)|(base.put_volume<cfg.n_options)).sum()),
                 max_stale_iv=int(base.stale_iv_days.max()),total_minute_bars=int(base.bars.sum()),
                 dte_min=int(base.dte.min()),dte_max=int(base.dte.max()),
                 tail_stress=stress)
    write_json(out/'summary.json',summary)
    return clean(summary)

if __name__=='__main__':
    summary=calculate()
    print(json.dumps(summary['metrics'],ensure_ascii=False,indent=2))
