"""Lightweight checks for the first MPCC / shared RacingEnv integration."""
import unittest
import numpy as np
from sim import RacingEnv
from controllers.mpc.run_racing_env import SharedTrackAdapter, controller_from_env, safe_stop


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.env = RacingEnv(car='numpy', track='training', log=False)
        _, self.info = self.env.reset(seed=0)

    def tearDown(self):
        self.env.close()

    def test_adapter_matches_shared_geometry(self):
        adapter = SharedTrackAdapter(self.env.track)
        ss = np.array([0., 1., 10., self.env.track.length + 2.])
        centers = adapter.center(ss)
        self.assertEqual(centers.shape, (4, 2))
        for i, s in enumerate(ss):
            np.testing.assert_allclose(centers[i], self.env.track.point_at(float(s)))
            self.assertAlmostEqual(adapter.curvature(ss)[i], self.env.track.curvature(float(s)))

    def test_shared_car_limits_and_stop(self):
        controller = controller_from_env(self.env, horizon=4)
        p = self.env.car_params
        self.assertAlmostEqual(controller.car.wheelbase, p['wheelbase'])
        self.assertAlmostEqual(controller.cfg.safety_margin, p['width'] / 2 + 0.03)
        np.testing.assert_allclose(safe_stop(self.env, self.info), [-1., 0.])

    def test_mpcc_runs_one_control_step(self):
        controller = controller_from_env(self.env, horizon=4)
        state = np.array([self.info[k] for k in ('x', 'y', 'psi', 'vx')])
        cmd, _, _ = controller.solve(state, self.info['progress'], self.info['delta'])
        action = self.env.to_action(*cmd)
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue((np.abs(action) <= 1).all())
        _, _, _, _, info = self.env.step(action)
        self.assertFalse(info['off_track'])


if __name__ == '__main__':
    unittest.main()
