"""Tests math/actuator interface without claiming real MuJoCo physics tests."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mujoco_bridge import yaw_from_quat_wxyz, MujocoMPCCBridge, BridgeConfig
from track import Track
from bicycle_model import CarParams

class FakeModel:
    opt = types.SimpleNamespace(timestep=0.005)
    actuator_ctrllimited = np.array([True, True])
    actuator_ctrlrange = np.array([[-0.42, 0.42], [-1.0, 1.0]])

class FakeData:
    def __init__(self, pos):
        self.time=0.
        self.xpos=np.array([[0.,0.,0.],[*pos,0.]])
        self.xquat=np.array([[1.,0.,0.,0.],[1.,0.,0.,0.]])
        self.ctrl=np.zeros(2)

class TestBridge(unittest.TestCase):
    def test_yaw(self):
        q=[np.cos(np.pi/4),0,0,np.sin(np.pi/4)]
        self.assertAlmostEqual(yaw_from_quat_wxyz(q),np.pi/2)

    def test_bridge_fallback_and_logging(self):
        fake_mj=types.SimpleNamespace(
            mjtObj=types.SimpleNamespace(mjOBJ_BODY=1,mjOBJ_ACTUATOR=2),
            mj_name2id=lambda model,typ,name: {'car':1,'steering':0,'drive':1}.get(name,-1))
        with patch.dict(sys.modules,{'mujoco':fake_mj}):
            track=Track(Path(__file__).parent/'training_track.csv')
            bridge=MujocoMPCCBridge(FakeModel(),FakeData(track.center(0)),track,
                                    BridgeConfig(),CarParams(max_speed=3.))
            bridge.controller.solve=lambda *args: (_ for _ in ()).throw(RuntimeError('forced'))
            self.assertTrue(bridge.tick())
            self.assertEqual(bridge.fallbacks,1)
            self.assertAlmostEqual(bridge.data.ctrl[1],-1.0)
            self.assertEqual(len(bridge.records),1)

if __name__=='__main__': unittest.main()
