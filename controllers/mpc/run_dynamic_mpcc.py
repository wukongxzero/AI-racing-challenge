"""Compare standalone dynamic-Pacejka MPCC with the existing kinematic baseline.

Run from repository root:
    python3 -m controllers.mpc.run_dynamic_mpcc --steps 900 --plot dynamic_mpcc.png
"""
import argparse
import time
from pathlib import Path
import numpy as np
from sim import RacingEnv
from sim.car_casadi import state_from_info
from .dynamic_mpcc import DynamicMPCC, DynamicConfig
from .run_racing_env import drive as drive_kinematic, controller_from_env, save_plot, safe_stop


def drive_dynamic(env, controller, steps=900, verbose=True):
    _,info=env.reset(seed=0)
    history=[(info['x'],info['y'])]
    timings=[]
    dynamic_steps=launch_steps=recoveries=fallbacks=0
    for i in range(steps):
        tic=time.perf_counter()
        # Below 1.5 m/s the simulator is partly kinematic: avoid ill-conditioned
        # slip-angle computations and launch with the proven kinematic MPCC.
        if info['vx'] < 1.5:
            launch_steps+=1
            state=np.array([info['x'],info['y'],info['psi'],max(0,info['vx'])])
            method=controller.launch
        else:
            dynamic_steps+=1
            state=state_from_info(info,'dynamic')
            method=controller
        try:
            u,_,_=method.solve(state,float(info['progress']),float(info['delta']))
            action=env.to_action(*map(float,u))
        except (RuntimeError,ValueError,FloatingPointError) as exc:
            # Retry from a fresh initialization once. A failed dynamic solve
            # uses braking (not unvalidated silent kinematic substitution).
            controller.guess_x=controller.guess_u=None
            controller.guess_lam=None
            if method is controller.launch:
                method.guess_x=method.guess_u=None
            try:
                u,_,_=method.solve(state,float(info['progress']),float(info['delta']))
                action=env.to_action(*map(float,u))
                recoveries+=1
            except (RuntimeError,ValueError,FloatingPointError):
                fallbacks+=1
                action=safe_stop(env,info)
                if verbose:
                    print(f'FALLBACK step={i}: {str(exc).splitlines()[0]}')
        timings.append((time.perf_counter()-tic)*1000)
        _,_,terminated,truncated,info=env.step(action)
        history.append((info['x'],info['y']))
        if verbose and (i%40==0 or info['lap_completed']):
            print(f"step={i:4d} t={info['t']:.2f}s vx={info['vx']:.2f} "
                  f"ec={info['e_c']:+.3f} solve={timings[-1]:.1f}ms "
                  f"off={info['off_track_count']} recovery={recoveries} fallback={fallbacks}")
        if terminated or truncated:
            break
    return {'steps':len(timings),'time':info['t'],'lap':info['lap'],
            'progress':info['progress'],'off_track_count':info['off_track_count'],
            'solver_recoveries':recoveries,'solver_fallbacks':fallbacks,
            'over_50ms':int(np.sum(np.asarray(timings)>50)),
            'mean_solve_ms':float(np.mean(timings)),
            'max_solve_ms':float(np.max(timings)),
            'p95_solve_ms':float(np.percentile(timings,95)),
            'p99_solve_ms':float(np.percentile(timings,99)),
            'history':np.array(history),'info':info,'dynamic_steps':dynamic_steps,
            'launch_steps':launch_steps}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track',choices=['training','hidden'],default='training')
    ap.add_argument('--steps',type=int,default=900)
    ap.add_argument('--horizon',type=int,default=8)
    ap.add_argument('--speed',type=float,default=3.0)
    ap.add_argument('--substeps',type=int,default=3)
    ap.add_argument('--dual-warm-start',action='store_true',help='Experimental IPOPT multiplier warm start')
    ap.add_argument('--fast',action='store_true',help='Use validated horizon=6 and RK4 substeps=2 preset; overrides --horizon/--substeps')
    ap.add_argument('--plot',type=Path)
    ap.add_argument('--log',type=Path)
    ap.add_argument('--baseline',action='store_true',help='Run original kinematic MPCC instead')
    args=ap.parse_args()
    if args.fast:
        args.horizon, args.substeps = 6, 2
    env=RacingEnv(car='numpy',track=args.track,off_track='grass',max_laps=1,
                  time_limit=args.steps*0.05,log=bool(args.log))
    try:
        if args.baseline:
            c=controller_from_env(env,speed_cap=args.speed)
            r=drive_kinematic(env,c,steps=args.steps,verbose=False)
        else:
            c=DynamicMPCC(env,DynamicConfig(horizon=args.horizon,speed=args.speed,substeps=args.substeps,dual_warm_start=args.dual_warm_start))
            r=drive_dynamic(env,c,steps=args.steps)
        print(f"RESULT steps={r['steps']} laps={r['lap']} t={r['time']:.2f}s "
              f"progress={r['progress']:.2f}/{env.track.length:.2f} "
              f"off_track={r['off_track_count']} fallback={r['solver_fallbacks']} "
              f"recoveries={r['solver_recoveries']} over_50ms={r['over_50ms']} "
              f"mean={r['mean_solve_ms']:.1f}ms max={r['max_solve_ms']:.1f}ms"
              + (f" p95={r['p95_solve_ms']:.1f}ms p99={r['p99_solve_ms']:.1f}ms" if 'p95_solve_ms' in r else ''))
        if not args.baseline:
            print(f"DYNAMIC_STEPS={r['dynamic_steps']} LAUNCH_STEPS={r['launch_steps']}")
        if args.plot:
            save_plot(env,r,args.plot)
            print(f'saved plot: {args.plot}')
        if args.log:
            env.save_log(str(args.log))
    finally:
        env.close()

if __name__=='__main__':
    main()
