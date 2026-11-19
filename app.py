"""Small local research UI. Binds to localhost only and never places market orders."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from pathlib import Path
from datetime import date
from dataclasses import replace
import json, html, webbrowser, threading, os
from engine import Config, snapshot, prepare
from analysis import calculate

ROOT=Path(__file__).resolve().parent
LABELS={'combined':'Стрэддл + грид','straddle':'Стрэддл','grid':'Грид','delta_hedged':'Дельта-хедж'}

def page(folder='results',notice=''):
    summary=json.loads((ROOT/folder/'summary.json').read_text(encoding='utf8'))
    cfg=json.loads((ROOT/folder/'config.json').read_text(encoding='utf8'))
    rows=''.join(f'<tr><td>{LABELS[m["strategy"]]}</td><td>{m["pnl"]:,.0f}</td><td>{m["sharpe"]:.2f}</td><td>{m["max_drawdown_pct"]:.2f}%</td><td>{m["cycles"]}</td></tr>' for m in summary['metrics'])
    def inp(label,key,step,minv,maxv):
        return f'<label>{label}<input type="number" name="{key}" value="{cfg[key]}" step="any" min="{minv}" max="{maxv}" required></label>'
    fields=inp('Шаг как доля премии стрэддла','step_fraction','.01','.03','.5')+inp('Лотов на уровень (0 = по гамме)','fixed_lots','1','0','10')+inp('Ширина диапазона × премия','width','.1','.25','2')
    fields+='<label>Целевой срок опциона<select name="target_dte">'+''.join(f'<option value="{v}" {"selected" if v==cfg["target_dte"] else ""}>{v} дней</option>' for v in [7,14,21,28])+'</select></label>'
    mode='adaptive' if cfg.get('adaptive_gamma') else 'boundary' if cfg.get('boundary_exit') else 'fixed'
    fields+='<label>Правило сетки<select name="mode">'+''.join(f'<option value="{v}" {"selected" if v==mode else ""}>{label}</option>' for v,label in [('fixed','Фиксированный объём'),('adaptive','Объём по текущей гамме'),('boundary','Закрыть инвентарь на границе')])+'</select></label>'
    for key,label in [('start','Начало'),('end','Конец')]:fields+=f'<label>{label}<input type="date" name="{key}" value="{cfg[key]}" min="2025-12-01" max="2026-10-30" required></label>'
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>FamilyBroker · Стрэддл и грид</title><style>
    body{{font:16px Aptos,Arial,sans-serif;color:#111;background:#f5f5f5;margin:0}}main{{max-width:1080px;margin:30px auto;background:white;padding:30px}}h1{{font-size:28px}}p{{line-height:1.5}}form{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;padding:20px;background:#f3f3f3}}label{{font-size:14px}}input,select{{box-sizing:border-box;width:100%;padding:10px;margin-top:6px;border:1px solid #999;border-radius:3px;background:white;font:inherit}}button{{background:#111;color:white;padding:13px;border:0;font:inherit;cursor:pointer}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:12px;border:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eee}}img{{width:100%;height:auto;margin:10px 0}}.note{{background:#eee;padding:12px}}a{{color:#222}}@media(max-width:700px){{main{{padding:16px;margin:0}}form{{grid-template-columns:1fr}}table{{font-size:12px}}th,td{{padding:6px}}}}</style>
    <main><h1>Стрэддл и грид на Si · версия 2</h1><p><a href="/">Исходный период</a> · <a href="/control">Контрольный период июль–октябрь</a></p><p>Опционы переоцениваются по модели с IV предыдущего дня. Результаты не подтверждают исполнимость сделок по расчётным ценам. Денежный счёт переносится между циклами; при нехватке обеспечения действует модельное принудительное закрытие.</p>
    <p class="note">{html.escape(notice) if notice else 'Базовый расчёт сохранён в results. Изменения параметров сохраняются отдельно в ui_results.'}</p>
    <form method="post" action="/run">{fields}<button type="submit" onclick="this.textContent='Считаю…';">Пересчитать</button></form>
    <p>20 стрэддлов; лимит позиции 20 фьючерсов; капитал 1 млн ₽. Вход во вторник, выход в пятницу. Фактический срок зависит от доступных серий. Остальные допущения — в README.</p>
    <table><tr><th>Стратегия</th><th>P&amp;L, ₽</th><th>Sharpe</th><th>Макс. просадка</th><th>Циклы</th></tr>{rows}</table>
    <h2>Капитал</h2><img src="/{folder}/equity.png" alt="Кривые капитала четырёх стратегий">
    <h2>Из чего сложился результат</h2><img src="/{folder}/decomposition.png" alt="Декомпозиция результата по грекам, гриду и издержкам">
    <h2>Циклы и волатильность</h2><img src="/{folder}/distribution_rv.png" alt="Распределение результата и отношение RV к IV">
    <h2>Три примера</h2>{''.join(f'<img src="/{folder}/cycle_{label}.png" alt="Цикл {label}">' for label in ['best','worst','typical'])}
    <p><a href="/{folder}/cycles.csv">Скачать результаты циклов CSV</a> · <a href="/{folder}/decomposition.csv">Скачать декомпозицию CSV</a> · <a href="/{folder}/minute_ledger.csv.gz" download>Поминутный журнал капитала (CSV.GZ)</a> · <a href="/{folder}/transactions.csv" download>Журнал сделок (CSV)</a></p></main></html>'''

class Handler(BaseHTTPRequestHandler):
    def send(self,body,status=200,mime='text/html; charset=utf-8'):
        if isinstance(body,str):body=body.encode('utf8')
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path=='/':return self.send(page())
        if self.path=='/control':return self.send(page('results_oos'))
        parts=self.path.strip('/').split('/')
        allowed={'equity.png','decomposition.png','distribution_rv.png','cycle_best.png','cycle_worst.png','cycle_typical.png','cycles.csv','decomposition.csv','minute_ledger.csv.gz','transactions.csv'}
        if len(parts)==2 and parts[0] in ['results','results_oos','ui_results'] and parts[1] in allowed:
            p=ROOT/parts[0]/parts[1]
            if p.is_file():
                mime={'.png':'image/png', '.gz':'application/gzip', '.csv':'text/csv; charset=utf-8'}[p.suffix]
                return self.send(p.read_bytes(),mime=mime)
        self.send('Не найдено',404)
    def do_POST(self):
        if self.path!='/run':return self.send('Не найдено',404)
        try:
            length=int(self.headers.get('Content-Length',0))
            if length>5000:raise ValueError('Слишком большой запрос')
            f={k:v[0] for k,v in parse_qs(self.rfile.read(length).decode()).items()}
            cfg=replace(Config(),step_fraction=float(f['step_fraction']),fixed_lots=int(f['fixed_lots']),
                        width=float(f['width']),target_dte=int(f['target_dte']),start=f['start'],end=f['end'],
                        adaptive_gamma=f.get('mode')=='adaptive',boundary_exit=f.get('mode')=='boundary')
            if not(.03<=cfg.step_fraction<=.5 and 0<=cfg.fixed_lots<=10 and .25<=cfg.width<=2 and cfg.target_dte in [7,14,21,28]):raise ValueError('Параметры вне допустимого диапазона')
            if not(date(2025,12,1)<=date.fromisoformat(cfg.start)<date.fromisoformat(cfg.end)<=date(2026,10,30)):raise ValueError('Проверьте даты')
            if cfg.start<'2026-07-01' and cfg.end>'2026-06-30':raise ValueError('Исходный и контрольный периоды пересчитываются отдельно')
            if f.get('mode','fixed') not in ['fixed','adaptive','boundary']:raise ValueError('Неизвестное правило сетки')
            os.environ['FAMILYBROKER_SNAPSHOT']='snapshot_oos.json.gz' if cfg.start>='2026-07-01' else 'snapshot.json.gz'
            snapshot.cache_clear();prepare.cache_clear()
            try:s=calculate(cfg,ROOT/'ui_results',with_sensitivity=False)
            finally:
                os.environ.pop('FAMILYBROKER_SNAPSHOT',None);snapshot.cache_clear();prepare.cache_clear()
            notice='Расчёт обновлён. Таблица и графики относятся к выбранным параметрам.'
            if any(m['capital_breaches'] for m in s['metrics']):notice+=' Отмечена нехватка обеспечения; проверьте журнал принудительных закрытий.'
            self.send(page('ui_results',notice))
        except (ValueError,KeyError) as e:self.send(page(notice=str(e)),400)

def main(open_browser=True):
    server=HTTPServer(('127.0.0.1',8765),Handler)
    print('Откройте http://127.0.0.1:8765 — остановка Ctrl+C',flush=True)
    if open_browser:threading.Timer(.5,lambda:webbrowser.open('http://127.0.0.1:8765')).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':
    import sys
    main('--no-browser' not in sys.argv)
