from __future__ import annotations
"""Strategy JSON loader and simulator runner.

Input JSON structure example:
{
  "FYs": {
     "FY1": {"speed": 120, "azimuth": 0.3, "bombs": [{"deploy_time":1.5, "explode_delay":3.6}]},
     "FY2": {"speed": 100, "aim_fake_target": true, "bombs": []},
     "FY3": {"speed": 110, "direction": [1,0,0], "bombs": [{"deploy_time":5, "explode_delay":4}]}
  },
  "Ms": ["M1","M2","M3"],
  "VERBOSE": true
}

We convert this into the decision dict used by api.decision and call simulate_with_decision.
"""
from pathlib import Path
from typing import Dict, Any
import json
from api.decision import simulate_with_decision

ALLOWED_FY = {f"FY{i}":None for i in range(1,10)}  # accept FY1..FY9 gracefully
ALLOWED_M = {f"M{i}":None for i in range(1,10)}


def load_strategy(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('strategy JSON 顶层必须为对象')
    fys = data.get('FYs') or data.get('fys')
    if isinstance(fys, list):
        # allow legacy list form: [ {"name":"FY1", ...}, ...]
        fys_dict = {}
        for item in fys:
            name = item.get('name')
            if not name:
                raise ValueError('FY list 元素缺少 name')
            fys_dict[name] = {k:v for k,v in item.items() if k!='name'}
        fys = fys_dict
    if not isinstance(fys, dict):
        raise ValueError('FYs 必须为对象 {"FY1": {...}, ...}')
    Ms = data.get('Ms') or data.get('ms') or ['M1']
    if not isinstance(Ms, (list, tuple)):
        raise ValueError('Ms 必须为列表')
    Ms = [str(m) for m in Ms]
    for m in Ms:
        if m not in ALLOWED_M:
            raise ValueError(f'未知导弹名称: {m}')
    verbose = bool(data.get('VERBOSE', True))
    # build decision
    drones = []
    for name, spec in fys.items():
        if name not in ALLOWED_FY:
            raise ValueError(f'未知无人机名称: {name}')
        # minimal fields
        if 'speed' not in spec:
            raise ValueError(f'{name} 缺少 speed')
        drone_entry = {
            'pos0': spec.get('pos0') or None,  # api.decision will verify / we fill default below
            'speed': spec['speed'],
        }
        if not drone_entry['pos0']:
            # use constants defaults for known names
            try:
                from simcore.constants import DRONES_POS0
                if name in DRONES_POS0:
                    drone_entry['pos0'] = DRONES_POS0[name].tolist()
                else:
                    raise KeyError
            except Exception:
                raise ValueError(f'{name} 未提供 pos0 且无法从常量推断')
        # heading specification priority: direction > azimuth > aim_fake_target
        if 'direction' in spec:
            drone_entry['direction'] = spec['direction']
        elif 'azimuth' in spec:
            drone_entry['azimuth'] = spec['azimuth']
        else:
            if spec.get('aim_fake_target', True):
                drone_entry['aim_fake_target'] = True
            else:
                raise ValueError(f'{name} 缺少 direction/azimuth 且 aim_fake_target 不为真')
        bombs = spec.get('bombs', [])
        drone_entry['bombs'] = bombs
        drones.append(drone_entry)
    decision = {'drones': drones}
    which = 'M1M2M3' if set(Ms)=={'M1','M2','M3'} else (Ms[0] if len(Ms)==1 else Ms)
    return dict(decision=decision, which=which, verbose=verbose)


def run_strategy(path: str | Path, *, dt: float=0.02, occlusion_method: str='sampling') -> Dict[str, Any]:
    s = load_strategy(path)
    res = simulate_with_decision(s['decision'], which=s['which'], dt=dt, occlusion_method=occlusion_method, verbose=s['verbose'])
    return res

__all__ = ['load_strategy','run_strategy']
