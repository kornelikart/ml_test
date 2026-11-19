"""Frozen comparisons; no parameter search. Run: python research_v2.py"""
from pathlib import Path
from dataclasses import replace
import os,json
import pandas as pd
from engine import Config,run,prepare,snapshot
from analysis import calculate,write_json

ROOT=Path(__file__).resolve().parent
VARIANTS=[('fixed_v2',{}),('adaptive_gamma',{'adaptive_gamma':True}),('exit_boundary',{'boundary_exit':True}),
          ('liquid_20',{'min_option_volume':20}),('spread_1pct',{'option_half_spread':.01}),
          ('spread_5pct',{'option_half_spread':.05}),('valuation_1845',{'valuation_clock':'18:45:00'})]
def research():
    comparisons=[]
    for period,filename,start,end,out in [
        ('development','snapshot.json.gz','2025-12-01','2026-06-30','results'),
        ('control','snapshot_oos.json.gz','2026-07-01','2026-10-30','results_oos')]:
        os.environ['FAMILYBROKER_SNAPSHOT']=filename;snapshot.cache_clear();prepare.cache_clear()
        cfg=replace(Config(),start=start,end=end)
        summary=calculate(cfg,ROOT/out,with_sensitivity=(period=='development'))
        print(period,'base',summary['metrics'][0]['pnl'],flush=True)
        for label,changes in VARIANTS:
            rs,sk=run(replace(cfg,**changes),detail=True,strategies=['combined'])
            cumulative=0;peak=cfg.capital;dd=0;minfree=cfg.capital
            for r in rs:
                for point in r['path']:
                    eq=point['equity'];peak=max(peak,eq);dd=max(dd,(peak-eq)/peak)
                cumulative+=r['pnl'];minfree=min(minfree,r['min_free_equity'])
            row=dict(period=period,variant=label,cycles=len(rs),pnl=cumulative,drawdown_pct=100*dd,
                wins=sum(r['pnl']>0 for r in rs),forced_liquidations=sum(r['forced_liquidation'] for r in rs),
                min_free_equity=minfree,skipped=len(sk))
            comparisons.append(row);print(row,flush=True)
        audit=dict(period=period,cycles=summary['metrics'][0]['cycles'],low_volume_cycles=summary['low_volume_cycles'],
                   max_stale_iv_days=summary['max_stale_iv'],excluded=summary['skipped'],
                   option_marks='model, based on prior daily settlement; NOT bid/ask',
                   valuation_clock_assumption=cfg.valuation_clock,available_after='next calendar date',
                   publication_timestamps_observed=False)
        write_json(ROOT/out/'data_audit.json',audit)
    pd.DataFrame(comparisons).to_csv(ROOT/'comparison_v2.csv',index=False,encoding='utf-8-sig')
    os.environ.pop('FAMILYBROKER_SNAPSHOT',None);snapshot.cache_clear();prepare.cache_clear()
    return comparisons

if __name__=='__main__':research()
