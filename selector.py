"""Blast Radius selector.

For every patch: map it to a conservative candidate test set (coverage + class + fixture maps built
from the pinned sqlglot commit), then either run the cheap candidates against the patched source and
print exactly the failing ones, or print the candidates split over workers when that is cheaper.
Never crashes: any internal error falls back to printing every test.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA = Path(os.environ.get("BLAST_DATA", "/data"))
SRC = os.environ.get("BLAST_SRC", "/opt/sqlglot")
MAP = os.environ.get("BLAST_MAP", str(HERE / "data"))
HEAVY = 3.0  # tests at least this long (recorded seconds) are printed, never run locally
RUN_MAX = 30.0  # never run more than this many recorded seconds of candidates


def evidence_match(heavy_id, test):
    """Cheap tests whose failure says a heavy test may fail too: same pipeline (executor/optimizer for the TPC
    tests), same class (parser, generator; generator also any dialect test)."""
    cls = test["id"].split("::")[1]
    if heavy_id.endswith("::test_max_nodes"):
        return cls == "TestParser"
    if heavy_id.endswith("::test_generator_memory"):
        return cls == "TestGenerator" or test["file"].startswith("tests/dialects/")
    return cls in ("TestExecutor", "TestOptimizer")


def keep_heavy(rest, failed, ev):
    """Heavy candidates the run's failures vouch for: one of their evidence tests failed, the run was too thin to
    judge, or the damage is wide (0 misses on 454 labeled commits for the class rule)."""
    failed = set(failed)
    if len(rest) < MIN_EVIDENCE or len(failed) >= MANY_FAIL:
        return set(ev)
    return {h for h, e in ev.items() if len(failed & e) >= EV_MIN}


MANY_FAIL = 50
# a single failing evidence test never came with a failing heavy test: tpcds 0 of 86 labeled commits
# (it failed in 37, always with 2+ evidence failures); tpch, max_nodes, generator_memory need far more
EV_MIN = 2


MIN_EVIDENCE = 2  # 0 heavy-only failures in 59 labeled commits whose cheap candidates passed
GEN_CORE = {"sqlglot/generator.py", "sqlglot/dialects/dialect.py", "sqlglot/generators/base.py"}
RUN_MUL = 1.1
RUN_FIX = 0.4
WIDE_COUNT = 0.5  # ... or when it already names this fraction of all tests
WIDE = 0.8  # print everything when the chosen set costs this fraction of printing everything
T0 = time.time()
PAR_AFTER = 70.0


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def pack(nums, ms):
    """Best split of nums over 1..8 workers (LPT per k), minimising max load + 2 s per worker."""
    if not nums:
        return [], 0
    items = sorted(set(nums), key=lambda n: -ms[n])
    best = None
    for k in range(1, min(8, len(items)) + 1):
        loads, ws = [0] * k, [[] for _ in range(k)]
        for n in items:
            i = loads.index(min(loads))
            ws[i].append(n)
            loads[i] += ms[n]
        cost = max(loads) + 2000 * k
        if best is None or cost < best[1]:
            best = (ws, cost)
    return best


def bits_to_nums(b):
    out, i = [], 0
    while b:
        low = b & -b
        out.append(low.bit_length() - 1)
        b ^= low
    return out


def main():
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    ms = [round(t["seconds"] * 1000) for t in tests]
    try:  # measured CPU per test under our runner (sequential, no worker pool); recorded seconds otherwise
        cpu_map = json.loads((Path(MAP) / "cpu.json").read_text())
    except Exception:
        cpu_map = {}
    try:
        canary_all = json.loads((Path(MAP) / "canary.json").read_text())
    except Exception:
        canary_all = []
    try:
        noise_ids = set(json.loads((Path(MAP) / "noise.json").read_text()))
    except Exception:
        noise_ids = set()
    noise = {t["number"] for t in tests if t["id"] in noise_ids}
    gm = next((t["number"] for t in tests if t["id"].endswith("TestGenerator::test_generator_memory")), None)
    cpu_est = [cpu_map.get(str(t["number"]), t["seconds"]) for t in tests]
    patches = sorted((DATA / "commits").glob("*.patch"))
    out = {}

    def emit(cid, workers):
        out[cid] = workers

    all_cost = pack(list(range(len(tests))), ms)[1]

    def emit_set(cid, nums):
        """Print `nums` packed over workers, or every test when that is nearly as cheap: a wide
        change can fail tests our per-test maps do not tie to it (shared-process contamination)."""
        w, c = pack(nums, ms)
        out[cid] = all_workers if c >= WIDE * all_cost or len(set(nums)) >= WIDE_COUNT * len(tests) else w

    all_workers = pack(list(range(len(tests))), ms)[0]
    try:
        import hashlib
        import mapper
        import patchlib
        import runner
        want = (Path(MAP) / "ids.sha1").read_text().strip()
        if hashlib.sha1("\n".join(t["id"] for t in tests).encode()).hexdigest() != want:
            raise RuntimeError("tests.json differs from the map's test list")
        mp = mapper.Mapper(MAP, SRC, tests)
        try:
            import cmpmap
            cm = cmpmap.CmpMap(MAP, len(tests))
        except Exception:
            cm = None
    except Exception:
        log(traceback.format_exc())
        for p in patches:
            print(json.dumps({"commit": p.stem, "workers": all_workers}))
        return

    def orig(path):
        try:
            with open(f"{SRC}/{path}", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    jobs_todo = []
    for p in patches:
        cid = p.stem
        try:
            fds = patchlib.parse(p.read_text(encoding="utf-8", errors="replace"))
            bits, notes = mp.candidates(fds, orig)
            proof = cm.affected(fds, orig, lambda fd, old: patchlib.apply(fd, old), mp.mentions) if cm and not os.environ.get("BLAST_NO_CMP") else None
            if proof is not None:  # only comparison operators/literals changed: just the tests whose outcome flips
                bits, mp.import_files, mp.assumed = (proof | (cm.unknown & bits if cm.had_edits else 0)) & mp.all_bits, set(), 0
            assumed = bits_to_nums(mp.assumed & mp.all_bits)
            bits &= ~mp.assumed
            nums = bits_to_nums(bits)
            check_files = {f for f in mp.import_files if f.startswith("sqlglot/")}
            for f in list(check_files):  # a renamed/removed name breaks the modules that import it
                check_files.update(mp.importer_files(f))
            check = sorted({mp.modname(f) for f in check_files if not f.endswith("__main__.py")})
            if gm is not None and gm in nums and not any(
                    fd.path in GEN_CORE or fd.old in GEN_CORE for fd in fds):
                nums.remove(gm)  # 0 failures in 33 labeled cases; see NOTES (only generator internals can leak)
            heavy = [n for n in nums if tests[n]["seconds"] >= HEAVY]
            rest = [n for n in nums if tests[n]["seconds"] < HEAVY]
            ev = {h: {n for n in rest if n not in noise and evidence_match(tests[h]["id"], tests[n])} for h in heavy}
            s = sum(cpu_est[n] for n in rest) + sum(
                cpu_map.get("class:%s::%s" % (f, c), 0.0)
                for f, c in {tuple(tests[n]["id"].split("::")[:2]) for n in rest})
            printed = pack(rest, ms)[1] / 1000
            # run when it is cheaper than printing (a run that finds nothing prints [] = 0): measured on
            # practice, preferring print overshot (+81 simulated for -53 CPU)
            run = bool(rest) and s <= RUN_MAX and (bool(heavy) or s * RUN_MUL + RUN_FIX <= printed)
            direct = nums + assumed if not run else None
            if run or check:
                patched = {}
                for fd in fds:
                    path = fd.old or fd.path
                    if fd.new is None:
                        patched[path] = None
                    elif path.endswith(".py") or path.startswith("tests/"):
                        patched[fd.path] = patchlib.apply(fd, orig(path))
                canary = [n for n in canary_all if n not in set(nums)] if check else []
                jobs_todo.append(dict(cid=cid, patched=patched, run=rest if run else [], heavy=heavy, assumed=assumed, canary=canary,
                                      nums=nums, check=check, direct=direct, ev=ev,
                                      extra={f: mp.import_fail_bits(f) for f in check_files}))
            else:
                emit_set(cid, direct if direct is not None else assumed)
            log(cid, len(nums), round(s, 2), "heavy" if heavy else "", "->", "run" if run else ("check" if check else "print"))
        except Exception:
            log(cid, traceback.format_exc())
            emit(cid, all_workers)

    if any(j["run"] for j in jobs_todo):
        for m in ("duckdb", "pandas", "numpy", "unittest", "pickle", "concurrent.futures"):
            try:  # preload once; forked children inherit the imports
                __import__(m)
            except Exception:
                pass
    budget = float(os.environ.get('BLAST_BUDGET', 4.5 * len(patches) * 0.7))
    running = []
    # one child at a time: two concurrent children cost ~8% more CPU (shared caches/SMT); go parallel only
    # when the batch is running long (limit is 3-4.5 minutes of wall time)
    fixed_nproc = os.environ.get('BLAST_NPROC')

    def fallback(j):
        extra = 0
        for v in j["extra"].values():
            extra |= v
        return j["nums"] + j["assumed"] + bits_to_nums(extra)

    while jobs_todo or running:
        nproc = int(fixed_nproc) if fixed_nproc else (1 if time.time() - T0 < PAR_AFTER else 2)
        while jobs_todo and len(running) < nproc:
            j = jobs_todo.pop(0)
            if time.time() - T0 > budget:
                emit_set(j["cid"], fallback(j))
                continue
            to_run = j["run"] + j["canary"]
            limits = {n: max(6.0, 5 * tests[n]["seconds"]) for n in to_run}
            try:
                job = runner.Job(j["cid"], f"/tmp/ov_{j['cid']}", SRC, j["patched"], tests, to_run, limits,
                                 time.time() + 80, failfast=[sorted(e) for e in j["ev"].values() if len(e) >= EV_MIN] if j["run"] else [], check=j["check"],
                                 smoke=bool(j["check"]), noise=noise, ev_min=EV_MIN)
                running.append((j, job))
            except Exception:
                log(traceback.format_exc())
                emit_set(j["cid"], fallback(j))
        for item in list(running):
            j, job = item
            r = job.poll()
            if r is None:
                continue
            running.remove(item)
            cid = j["cid"]
            log(cid, "ran", len(j["run"]), "failed", len(r.get("failed", [])), "cpu", r.get("cpu"), r.get("importerr", ""),
                [tests[n]["id"].split("::", 1)[1] for n in r.get("failed", [])[:3]])
            if not r.get("ok"):
                log(cid, "run failed", r.get("err"))
                emit_set(cid, fallback(j))
            elif r.get("importerr"):
                extra = 0
                for f in j["extra"]:
                    if mp.modname(f) in r["importerr"]:
                        extra |= j["extra"][f]
                emit_set(cid, j["nums"] + j["assumed"] + bits_to_nums(extra))
            elif r.get("smoke_fail") or (set(r.get("failed", [])) & set(j["canary"])):
                log(cid, "canary alarm", r.get("smoke_fail"), sorted(set(r.get("failed", [])) & set(j["canary"]))[:5])
                # wide blast radius: everything that loads the failing dialects / lives in the failing canary files
                extra = 0
                for f in j["extra"].values():
                    extra |= f
                for d in r.get("smoke_fail") or []:
                    for k in ("dialects", "generators", "parsers"):
                        rel = f"sqlglot/{k}/{d}.py"
                        if rel in mp.closure or os.path.exists(f"{SRC}/{rel}"):
                            extra |= mp.import_fail_bits(rel)
                    extra |= mp.by_file.get(f"tests/dialects/test_{d}.py", 0)
                if "<import>" in (r.get("smoke_fail") or []):
                    extra = mp.all_bits
                for n in set(r.get("failed", [])) & set(j["canary"]):
                    extra |= mp.by_file.get(tests[n]["file"], 0)
                emit_set(cid, j["nums"] + j["assumed"] + bits_to_nums(extra))
            elif not j["run"]:
                emit_set(cid, j["direct"])
            else:
                failed = [n for n in r["failed"] if n not in set(j["canary"])]
                kept = sorted(keep_heavy(j["run"], [n for n in failed if n not in noise], j["ev"]))
                # stopped: every heavy with evidence is kept, so the unrun cheap tests ride along for free
                emit_set(cid, (j["run"] if r.get("stopped") else failed) + j["assumed"] + kept)
        time.sleep(0.005)
    for p in patches:
        print(json.dumps({"commit": p.stem, "workers": out.get(p.stem, all_workers)}))
    log("elapsed", round(time.time() - T0, 2))
    try:
        for ln in open("/sys/fs/cgroup/cpu.stat"):
            if ln.startswith("usage_usec"):
                log("cpu_seconds", int(ln.split()[1]) / 1e6)
    except OSError:
        pass


if __name__ == "__main__":
    main()
