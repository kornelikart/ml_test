"""Accounting, causality and risk invariants, using the included market snapshot."""
import unittest
import copy
from dataclasses import replace
import numpy as np
from engine import Config, black, implied, prepare, prepare_snapshot, run, simulate, snapshot

class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=Config();cls.rs,cls.sk=run(cls.cfg,detail=True)

    def test_greeks_and_inversion(self):
        F,K,t,v=80000.,80000.,14/365,.2; p,d,g,theta,vega=black(F,K,t,v)
        self.assertAlmostEqual(implied(F,K,t,p),v,places=9)
        self.assertAlmostEqual(float((black(F+1,K,t,v)[0]-black(F-1,K,t,v)[0])/2),float(d),places=6)
        self.assertAlmostEqual(float(black(F+1,K,t,v)[0]-2*p+black(F-1,K,t,v)[0]),float(g),places=6)
        self.assertAlmostEqual(float((black(F,K,t,v+.00001)[0]-black(F,K,t,v-.00001)[0])/.00002),float(vega),places=3)

    def test_all_ledgers_reconcile(self):
        for r in self.rs:
            greeks=sum(r[k] for k in ['delta_pnl','gamma_pnl','theta_pnl','vega_pnl','residual_pnl'])
            self.assertAlmostEqual(greeks,r['option_pnl'],places=6)
            cash=-sum(t['qty']*t['price'] for t in r['transactions'])
            self.assertAlmostEqual(cash,r['futures_pnl'],places=6)
            net=r['option_pnl']+r['futures_pnl']-sum(r[k] for k in ['futures_fees','option_fees','option_spread','slippage','capital_cost','funding'])
            self.assertAlmostEqual(net,r['pnl'],places=6)
            for point in r['path']:
                expected = r['starting_equity'] + point['option_pnl'] + point['grid_pnl'] - point['costs']
                self.assertAlmostEqual(point['equity'], expected, places=6)
            self.assertAlmostEqual(r['path'][-1]['equity'], r['ending_equity'], places=6)
            self.assertEqual(sum(t['qty'] for t in r['transactions']),0)

    def test_risk_and_time_invariants(self):
        for r in self.rs:
            self.assertLessEqual(r['peak_inventory'],self.cfg.inventory_cap)
            self.assertTrue(all(abs(t['inventory'])<=self.cfg.inventory_cap for t in r['transactions']))
            self.assertEqual(r['path'][-1]['inventory'],0)
            pre,_=prepare(r['cycle'],14)
            for ts,iv in zip(pre['ts'],pre['iv']):
                known=[m for m in pre['ivmarks'] if m[0]<ts.strftime('%Y-%m-%d')]
                self.assertAlmostEqual(iv,known[-1][1])

    def test_no_future_price_changes_past_orders(self):
        p,_=prepare('2025-12-02',14)
        changed=dict(p);changed['rows']=[dict(x) for x in p['rows']]
        # Change prices after the cutoff without altering any previous observation.
        cut=1000
        for row in changed['rows'][cut:]:
            for key in ['open','high','low','close']:row[key]+=5000
        a=simulate(p,self.cfg,'combined',True);b=simulate(changed,self.cfg,'combined',True)
        cutoff=p['ts'][cut].isoformat(' ')
        self.assertEqual([x for x in a['transactions'] if x['time']<=cutoff],
                         [x for x in b['transactions'] if x['time']<=cutoff])

    def test_cost_monotonicity(self):
        p,_=prepare('2025-12-02',14)
        a=simulate(p,self.cfg,'combined');b=simulate(p,replace(self.cfg,option_half_spread=.05),'combined')
        self.assertLess(b['pnl'],a['pnl'])
        self.assertAlmostEqual(a['pnl']-b['pnl'],b['option_spread']-a['option_spread'],places=6)

    def test_combined_equals_separate_legs(self):
        # No margin netting is assumed, so linear composition must reconcile.
        for cid in {r['cycle'] for r in self.rs}:
            d={r['strategy']:r for r in self.rs if r['cycle']==cid}
            self.assertAlmostEqual(d['combined']['pnl'],d['straddle']['pnl']+d['grid']['pnl'],places=6)

    def test_cash_carries_between_cycles(self):
        for strategy in ['combined','straddle','grid','delta_hedged']:
            capital=self.cfg.capital
            for r in [x for x in self.rs if x['strategy']==strategy]:
                self.assertAlmostEqual(r['starting_equity'],capital)
                capital+=r['pnl'];self.assertAlmostEqual(r['ending_equity'],capital)

    def test_post_only_and_session_delay(self):
        for r in self.rs:
            if r['strategy'] not in ['grid','combined']:continue
            pre,_=prepare(r['cycle'],14)
            lookup={t.isoformat(' '):i for i,t in enumerate(pre['ts'])}
            for trade in r['transactions']:
                if trade['kind']!='maker':continue
                i=lookup[trade['time']]
                self.assertGreater(i,1)
                self.assertEqual(pre['rows'][i-1]['begin'][:10],pre['rows'][i-2]['begin'][:10])
                if trade['qty']>0:self.assertLess(trade['price'],pre['F'][i-1])
                else:self.assertGreater(trade['price'],pre['F'][i-1])

    def test_margin_failure_closes_next_open(self):
        pre,_=prepare('2025-12-02',14)
        r=simulate(pre,self.cfg,'combined',True,starting_equity=1000)
        self.assertTrue(r['forced_liquidation'])
        self.assertAlmostEqual(r['path'][-1]['equity'], r['ending_equity'], places=6)
        self.assertGreater(r['path'][-1]['time'],r['risk_trigger'])
        self.assertEqual(r['path'][-1]['price'],pre['rows'][1]['open'])
        self.assertEqual(r['path'][-1]['inventory'],0)
        self.assertAlmostEqual(-sum(t['qty']*t['price'] for t in r['transactions']),r['futures_pnl'])
        self.assertAlmostEqual(sum(r[k] for k in ['delta_pnl','gamma_pnl','theta_pnl','vega_pnl','residual_pnl']),r['option_pnl'])
        rs,sk=run(replace(self.cfg,capital=1000),strategies=['combined'])
        self.assertFalse(rs);self.assertTrue(any('equity' in x['reason'] for x in sk))

    def test_future_source_data_cannot_change_past(self):
        original=snapshot();changed=copy.deepcopy(original);cut='2025-12-04'
        for key,rows in changed['options'].items():
            for row in rows:
                if row['TRADEDATE']>=cut and row.get('SETTLEPRICE') is not None:row['SETTLEPRICE']*=1.2
        for rows in changed['futures'].values():
            for row in rows:
                if row['TRADEDATE']>=cut and row.get('SETTLEPRICE'):row['SETTLEPRICE']*=1.03
        for days in changed['bars'].values():
            for day,rows in days.items():
                if day>=cut:
                    for row in rows:
                        for k in ['open','close','high','low']:row[k]*=1.03
        a,_=prepare_snapshot(original,'2025-12-02',14);b,_=prepare_snapshot(changed,'2025-12-02',14)
        mask=np.array([t.strftime('%Y-%m-%d')<cut for t in a['ts']])
        for key in ['F','iv','p','delta','gamma']:np.testing.assert_allclose(a[key][mask],b[key][mask])
        ra=simulate(a,self.cfg,'combined',True);rb=simulate(b,self.cfg,'combined',True)
        self.assertEqual([t for t in ra['transactions'] if t['time'][:10]<cut],[t for t in rb['transactions'] if t['time'][:10]<cut])

    def test_invalid_option_marks_rejected(self):
        self.assertIsNone(implied(80000,78000,14/365,1999.9))
        self.assertIsNone(implied(80000,78000,14/365,float('nan')))

if __name__=='__main__':unittest.main(verbosity=2)
