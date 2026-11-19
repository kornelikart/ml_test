"""One-command entry point; missing packages go into an isolated local environment."""
from pathlib import Path
import importlib.util, subprocess, sys, os, venv
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
packages=['numpy','scipy','pandas','matplotlib','requests']
if not all(importlib.util.find_spec(x) for x in packages):
    env=ROOT/'.venv';py=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not py.exists():venv.create(env,with_pip=True)
    subprocess.check_call([str(py),'-m','pip','install','-r',str(ROOT/'requirements.txt')])
    raise SystemExit(subprocess.call([str(py),str(__file__),*sys.argv[1:]]))
if '--batch' in sys.argv:
    from analysis import calculate
    result=calculate();print('Готово: results/summary.json')
elif '--test' in sys.argv:
    raise SystemExit(subprocess.call([sys.executable,str(ROOT/'tests.py')]))
else:
    from app import main
    main('--no-browser' not in sys.argv)
