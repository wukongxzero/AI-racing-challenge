"""Tests for dynamic MPCC and team-model connection.
Run: python3 -m unittest controllers.mpc.test_dynamic_mpcc
"""
import unittest
import numpy as np
from sim import RacingEnv
from sim.car_casadi import state_from_info, dynamic_model
from sim.car_numpy import NumpyCar, CarParams
from .dynamic_mpcc import DynamicMPCC, DynamicConfig
from .run_dynamic_mpcc import drive_dynamic


class DynamicMPCCTests(unittest.TestCase):
    def test_model_matches_numpy_rhs(self):
        p=CarParams()
        car=NumpyCar(p)
        f=dynamic_model(p)
        state=np.array([2.,3.,.2,2.5,.05,.1])
        u=np.array([.2,.1])
        numpy_rhs=car._dynamic([*state,u[1]],u[0],0.0,p.mu)[:6]
        np.testing.assert_allclose(np.asarray(f(state,u)).reshape(-1),numpy_rhs,atol=1e-10)

    def test_env_state_shape(self):
        env=RacingEnv(car='numpy',track='training',log=False)
        try:
            _,info=env.reset(seed=0)
            self.assertEqual(len(state_from_info(info,'dynamic')),6)
        finally:
            env.close()

    def test_optional_dual_warm_start(self):
        env = RacingEnv(car='numpy', track='training', log=False,
                        max_laps=1, time_limit=4.0)
        try:
            ctrl = DynamicMPCC(env, DynamicConfig(horizon=6, substeps=3,
                                                  dual_warm_start=True))
            result = drive_dynamic(env, ctrl, steps=80, verbose=False)
            self.assertEqual(result['off_track_count'], 0)
            self.assertEqual(result['solver_fallbacks'], 0)
            self.assertGreater(result['dynamic_steps'], 50)
            self.assertIsNotNone(ctrl.guess_lam)
        finally:
            env.close()

    def test_controller_short_run(self):
        env=RacingEnv(car='numpy',track='training',log=False,max_laps=1,time_limit=4.0)
        try:
            ctrl=DynamicMPCC(env,DynamicConfig(horizon=6,substeps=3))
            result=drive_dynamic(env,ctrl,steps=80,verbose=False)
            self.assertEqual(result['off_track_count'],0)
            self.assertGreater(result['dynamic_steps'],50)
            self.assertEqual(result['solver_fallbacks'],0)
        finally:
            env.close()


if __name__=='__main__':
    unittest.main()
