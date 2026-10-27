"""The shared RolloutBackend contract on MockBackend (the reference implementation) and on
IsaacBackend over the fake Arena runner and fake policy server."""

from __future__ import annotations

from ladder.backend import MockBackend
from router.schemas import CodeRef

from tests.contracts.backend_contract import BackendUnderTest, RolloutBackendContract
from tests.helpers import DARK, NOMINAL, POLICY, make_backend

SEEDS = (20003, 5, 10001, 2, 0, 10000, 7)          # three builds, unsorted


class TestMockBackendContract(RolloutBackendContract):
    def make(self, tmp_path):
        return BackendUnderTest(MockBackend([], default_rate=0.6), NOMINAL, DARK, CodeRef("c0"), POLICY, SEEDS)


class TestIsaacBackendContract(RolloutBackendContract):
    def make(self, tmp_path):
        rig = make_backend(tmp_path)
        return BackendUnderTest(rig.backend, NOMINAL, DARK, rig.code(), POLICY, SEEDS)
