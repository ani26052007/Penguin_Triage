import sys
from statistics import median
from router.prereq.trace import EpisodeTrace, InMemoryTraceSource
from router.schemas import ActorKind, ActorRef, CodeRef, Incident, Mode, SceneSpec, TrainingSpec
from router.stage1.nominal import NominalStats
from router.stage1.obs_contract import ObservationContractCheck, metric, observation_metrics, opponent, required_vectors
from vla_isaac.traces.instrument_reader import read_instrument_dir

CAMERA = "ego_view"
CODE = CodeRef("pw@injected", {"client": {}})
POLICY = ActorRef(ActorKind.POLICY, "ckpt")
SPEC = TrainingSpec("gr00t_n1d6_gr1_ranch_bottle_into_fridge", image_size=(512, 512), channel_order="RGB")

def traces(d):
    eps = read_instrument_dir(d)
    return [EpisodeTrace(seed=i, mode=Mode.CLEAN, frames=ep.frames, step_pairs=ep.step_pairs,
                         raw_actions=ep.raw_actions, code_key=CODE.key())
            for i, (_, ep) in enumerate(sorted(eps.items()))]

def obs_nominal(clean, cameras=(CAMERA,), rel_floor=0.05, abs_floor=0.02):   # verbatim from tests/bugs_support.py
    vectors = {}
    for cam in cameras:
        frames = [fs.policy_input for t in clean for fs in t.frames if fs.camera == cam]
        vectors[metric(cam, "opp")] = tuple(median(c) for c in zip(*(opponent(f.channel_means) for f in frames)))
        vectors[metric(cam, "row_profile")] = tuple(median(c) for c in zip(*(f.row_profile for f in frames)))
        vectors[metric(cam, "col_profile")] = tuple(median(c) for c in zip(*(f.col_profile for f in frames)))
    assert set(vectors) == set(required_vectors(cameras))
    samples = observation_metrics(clean, cameras, vectors)
    floors = {k: max(rel_floor * abs(sum(v) / len(v)), abs_floor) for k, v in samples.items()}
    return NominalStats.from_samples(samples, vectors=vectors, min_std=floors)

def check(nominal, tr):
    refs = {"clean": "traces/clean"}
    src = InMemoryTraceSource({refs["clean"]: list(tr)})
    inc = Incident("real-gpu", SceneSpec("gr1_fridge_ranch"), SPEC, CODE, POLICY, (0,), Mode.CLEAN, log_refs=refs)
    return ObservationContractCheck(nominal, src, [CAMERA]).run(inc)

clean, bugged = traces(sys.argv[1]), traces(sys.argv[2])
print("episodes: clean", len(clean), "bugged", len(bugged), "| frames/episode", len(clean[0].frames))
nom = obs_nominal(clean)
for name, tr in (("CLEAN code (checked against its own nominal)", clean), ("BUG-02 code", bugged)):
    ev = check(nom, tr)
    print(f"\n== {name}: {len(ev)} evidence item(s)")
    for e in ev:
        print(f"  source={e.source} signal={e.signal} id={e.id} location={e.location} strength={e.strength:.2f}\n    {e.detail}")
