"""Reproducible high-speed Dynamic MPCC test with slip and tracking telemetry.

Run from repository root:
  python3 -m controllers.mpc.benchmark_high_speed --speeds 6 7 8 --output high_speed.csv

Note: slip angle is atan2(vy,vx) of the *simulated body velocity*, not
an independently measured tire slip angle. Solver times are wall-clock times.
"""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
from sim import RacingEnv
from sim.car_casadi import state_from_info
from .dynamic_mpcc import DynamicConfig, DynamicMPCC
from .run_racing_env import safe_stop


def run_case(speed, *, track='training', steps=950, horizon=6, substeps=2, seed=0):
    env = RacingEnv(car='numpy', track=track, off_track='grass',
                    max_laps=1, time_limit=steps * .05, log=False)
    try:
        ctrl = DynamicMPCC(env, DynamicConfig(horizon=horizon, substeps=substeps, speed=speed))
        _, info = env.reset(seed=seed)
        rows=[]
        recoveries=fallbacks=0
        for i in range(steps):
            start=time.perf_counter()
            launch=info['vx'] < 1.5
            if launch:
                state=np.array([info['x'],info['y'],info['psi'],max(0,info['vx'])])
                solver=ctrl.launch
            else:
                state=state_from_info(info, 'dynamic')
                solver=ctrl
            try:
                u,_,_=solver.solve(state,float(info['progress']),float(info['delta']))
                action=env.to_action(*map(float,u))
            except (RuntimeError,ValueError,FloatingPointError):
                ctrl.guess_x=ctrl.guess_u=ctrl.guess_lam=None
                if launch:
                    solver.guess_x=solver.guess_u=None
                try:
                    u,_,_=solver.solve(state,float(info['progress']),float(info['delta']))
                    action=env.to_action(*map(float,u))
                    recoveries+=1
                except (RuntimeError,ValueError,FloatingPointError):
                    action=safe_stop(env,info)
                    fallbacks+=1
            solve_ms=(time.perf_counter()-start)*1000
            _,_,terminated,truncated,info=env.step(action)
            beta=float(np.arctan2(info['vy'], max(info['vx'],1e-6)))
            rows.append({'speed_cap':speed,'step':i,'t':float(info['t']),
                         'vx':float(info['vx']),'vy':float(info['vy']),
                         'yaw_rate':float(info['yaw_rate']),
                         'beta_deg':float(np.degrees(beta)),
                         'e_c':float(info['e_c']),
                         'progress':float(info['progress']),
                         'solve_ms':solve_ms,'off_track':int(info['off_track']),
                         'fallback_total':fallbacks,'recovery_total':recoveries,
                         'launch':int(launch)})
            if terminated or truncated: break
        timings=np.array([r['solve_ms'] for r in rows])
        cross=np.array([abs(r['e_c']) for r in rows])
        slips=np.array([abs(r['beta_deg']) for r in rows])
        summary={'speed_cap':speed,'laps':int(info['lap']), 'lap_time_s':float(info['t']),
                 'steps':len(rows),'off_track_count':int(info['off_track_count']),
                 'fallbacks':fallbacks,'recoveries':recoveries,
                 'mean_solver_ms':float(timings.mean()),
                 'p99_solver_ms':float(np.percentile(timings,99)),
                 'max_solver_ms':float(timings.max()),
                 'over_50ms':int(np.count_nonzero(timings>50)),
                 'max_abs_ec_m':float(cross.max()),
                 'p95_abs_ec_m':float(np.percentile(cross,95)),
                 'max_abs_body_slip_deg':float(slips.max()),
                 'p95_abs_body_slip_deg':float(np.percentile(slips,95)),
                 'max_vx_mps':float(max(r['vx'] for r in rows))}
        return summary, rows
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--speeds',nargs='+',type=float,default=[6,7,8])
    parser.add_argument('--track',choices=['training','hidden'],default='training')
    parser.add_argument('--steps',type=int,default=950)
    parser.add_argument('--horizon',type=int,default=6)
    parser.add_argument('--substeps',type=int,default=2)
    parser.add_argument('--output',type=Path,default=Path('high_speed.csv'))
    args=parser.parse_args()
    summaries=[]
    for speed in args.speeds:
        summary, rows=run_case(speed, track=args.track,steps=args.steps,
                               horizon=args.horizon, substeps=args.substeps)
        summaries.append(summary)
        print('RESULT', ' '.join(f'{k}={v:.3f}' if isinstance(v,float) else f'{k}={v}'
                                 for k,v in summary.items()), flush=True)
        dest=args.output.with_name(args.output.stem+f'_{speed:g}ms_steps.csv')
        with dest.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=summaries[0].keys())
        writer.writeheader(); writer.writerows(summaries)
    print('SAVED', args.output, flush=True)

if __name__=='__main__':
    main()
