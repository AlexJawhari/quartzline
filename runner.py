"""Run chosen tests against a patched copy of sqlglot in a forked child and report which fail.

The patched tree is an overlay of mostly symlinks into the pristine checkout, so spawned
multiprocessing workers (which inherit sys.path) see the patched code too.
"""
import json
import os
import signal
import sys
import time
import unittest

import patchlib


class _Timeout(BaseException):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def build_overlay(root, src, patched):
    """patched: {rel path: text or None (deleted)}. Directories on a changed path are real,
    everything else is a symlink into src."""
    real = {""}
    for rel in patched:
        parts = rel.split("/")
        real |= {"/".join(parts[:i]) for i in range(1, len(parts))}

    def fill(rel):
        d = os.path.join(root, rel)
        os.makedirs(d, exist_ok=True)
        sd = os.path.join(src, rel)
        for name in os.listdir(sd) if os.path.isdir(sd) else ():
            r = f"{rel}/{name}" if rel else name
            if r in real:
                fill(r)
            else:
                os.symlink(os.path.join(sd, name), os.path.join(d, name))

    fill("")
    for d in sorted(real):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for rel, text in patched.items():
        dst = os.path.join(root, rel)
        if os.path.lexists(dst):
            os.unlink(dst)
        if text is not None:
            with open(dst, "w", encoding="utf-8", newline="") as f:
                f.write(text)


class _Inline:
    """Synchronous stand-in for ProcessPoolExecutor: spawned workers cost far more CPU than the
    work itself (each re-imports duckdb/pandas), and the tests only need the results."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        if initializer:
            initializer(*initargs)

    def submit(self, fn, *a, **k):
        import concurrent.futures as cf
        import pickle
        f = cf.Future()
        try:  # round-trip through pickle like a real worker (sqlglot pickles via serde)
            fn, a, k = pickle.loads(pickle.dumps((fn, a, k)))
            f.set_result(pickle.loads(pickle.dumps(fn(*a, **k))))
        except Exception as e:
            try:
                e = pickle.loads(pickle.dumps(e))
            except Exception:
                pass
            f.set_exception(e)
        return f

    def map(self, fn, *its, **kw):
        return map(fn, *its)

    def shutdown(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_TPC_USERS = {"test_optimized_tpch", "test_execute_tpch", "test_execute_tpcds"}


def _exec_stub(cls):
    cls.cache = {}
    cls.tpch_sqls = cls.tpcds_sqls = []
    cls.tpch_tables = cls.tpcds_tables = {}


def _smoke():
    """Load every dialect and run one trivial statement through it: class-body code that no longer
    works (e.g. a parsed SQL template built at import) breaks dialect loading for many tests."""
    bad = []
    try:
        import sqlglot
        from sqlglot.dialects import MODULE_BY_DIALECT
        names = sorted(MODULE_BY_DIALECT.values())
    except BaseException:
        return ["<import>"]
    for d in names:
        try:
            signal.setitimer(signal.ITIMER_PROF, 10)
            if d in ("dax", "prql"):  # not SQL-in dialects: only check that they load
                sqlglot.Dialect.get_or_raise(d).generator()
            else:
                sqlglot.parse_one("SELECT a, b FROM t WHERE x = 1 ORDER BY a", read=d).sql(d)
        except BaseException:
            bad.append(d)
        signal.setitimer(signal.ITIMER_PROF, 0)
    return bad


def child_main(wfd, root, src, patched, tests, nums, limits, failfast=False, check=(), smoke=False, noise=(), ev_min=1):
    """Runs in the forked child; writes a JSON result line to wfd and exits."""
    res = {"failed": [], "ok": True}
    try:
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        _t = [time.process_time()]
        def lap(name):
            if os.environ.get("BLAST_DEBUG"):
                now = time.process_time()
                print(f"  [{os.path.basename(root)}] {name} {now - _t[0]:.3f}", file=sys.stderr)
                _t[0] = now
        import gc
        gc.freeze()
        gc.set_threshold(50000, 20, 100)  # fewer collections: ~3% less CPU, memory stays far below the cap
        build_overlay(root, src, patched)
        # the stub is only valid while neither the executor tests nor the TPC fixtures are edited
        lazy_exec = not any(p.startswith(("tests/test_executor", "tests/fixtures/optimizer/tpc")) for p in patched)
        lap("overlay")
        sys.path.insert(0, root)
        os.environ["PYTHONPATH"] = root
        import concurrent.futures as cf
        cf.ProcessPoolExecutor = _Inline
        signal.signal(signal.SIGPROF, _alarm)
        import importlib
        bad = []
        for modname in check:  # a changed module that no longer imports breaks everything that loads it
            try:
                signal.setitimer(signal.ITIMER_PROF, 30)
                importlib.import_module(modname)
            except BaseException:
                bad.append(modname)
            signal.setitimer(signal.ITIMER_PROF, 0)
        lap("check imports")
        if bad:
            res["importerr"] = bad
            nums = []
        elif smoke:
            res["smoke_fail"] = _smoke()
            lap("smoke")
        by_cls = {}
        for n in sorted(nums, key=lambda n: tests[n]["id"]):
            f, c, m = tests[n]["id"].split("::")
            by_cls.setdefault((f, c), []).append((m, n))
        failed = set()
        sets = [set(s) for s in failfast or ()]

        def _satisfied():  # every heavy candidate already has a failing evidence test: nothing left to learn
            return bool(sets) and all(len(failed & s) >= ev_min for s in sets)
        order = sorted(by_cls.items(), key=lambda kv: sum(tests[n]["seconds"] for _, n in kv[1]))
        for (f, c), items in order:
            try:
                signal.setitimer(signal.ITIMER_PROF, 30)
                import sqlglot  # noqa: F401  (timed separately)
                lap("import sqlglot")
                cls = getattr(importlib.import_module(f[:-3].replace("/", ".")), c)
                signal.setitimer(signal.ITIMER_PROF, 0)
                lap("import " + f)
            except BaseException:
                signal.setitimer(signal.ITIMER_PROF, 0)
                failed.update(n for _, n in items)  # an unimportable file fails its tests
                continue
            byname = {m: n for m, n in items}

            class R(unittest.TestResult):
                def startTest(self, test):
                    super().startTest(test)
                    signal.setitimer(signal.ITIMER_PROF, limits.get(byname.get(test._testMethodName), 10))

                def stopTest(self, test):
                    signal.setitimer(signal.ITIMER_PROF, 0)
                    super().stopTest(test)

                def _mark(self, test):
                    n = byname.get(getattr(test, "_testMethodName", None))
                    if n is None:  # setUpClass / tearDownClass / module fixture failure
                        failed.update(byname.values())
                    else:
                        failed.add(n)
                    if _satisfied():
                        self.stop()

                def addFailure(self, test, err):
                    self._mark(test)

                def addError(self, test, err):
                    self._mark(test)

                def addSubTest(self, test, subtest, err):
                    if err is not None:
                        self._mark(test)

                def addUnexpectedSuccess(self, test):
                    self._mark(test)

            if c == "TestExecutor" and lazy_exec and not (set(byname) & _TPC_USERS):
                cls.setUpClass = classmethod(_exec_stub)  # skip loading the TPC-H/DS tables nobody here reads
                cls.tearDownClass = classmethod(lambda k: None)
            suite = unittest.TestSuite([cls(m) for m, _ in items])
            try:
                suite.run(R())
            except BaseException:
                failed.update(byname.values())
            signal.setitimer(signal.ITIMER_PROF, 0)
            lap("run " + c)
            if _satisfied():
                break
        res["failed"] = sorted(failed)
        res["stopped"] = _satisfied()
    except BaseException as e:  # any internal failure: caller falls back to printing candidates
        res = {"failed": [], "ok": False, "err": repr(e)[:200]}
    try:
        os.write(wfd, json.dumps(res).encode())
    finally:
        os._exit(0)


class Job:
    _n = 0

    def __init__(self, cid, root, src, patched, tests, nums, limits, deadline, failfast=False, check=(), smoke=False, noise=(), ev_min=1):
        self.cid, self.nums, self.deadline = cid, nums, deadline
        Job._n += 1
        r, w = os.pipe()
        self.pid = os.fork()
        if self.pid == 0:
            os.close(r)
            try:  # one CPU per child: idle library thread pools (duckdb, BLAS) cannot spread charged CPU
                cpus = sorted(os.sched_getaffinity(0))
                os.sched_setaffinity(0, {cpus[Job._n % len(cpus)]})
            except (AttributeError, OSError, IndexError):
                pass
            child_main(w, root, src, patched, tests, nums, limits, failfast, check, smoke, noise, ev_min)
        os.close(w)
        self.fd, self.buf, self.root = r, b"", root

    def poll(self):
        """Returns result dict when done, None while running."""
        import select
        while True:
            rd, _, _ = select.select([self.fd], [], [], 0)
            if not rd:
                break
            chunk = os.read(self.fd, 65536)
            if not chunk:
                break
            self.buf += chunk
        try:
            pid, _, ru = os.wait4(self.pid, os.WNOHANG)
            if pid:
                self.cpu = ru.ru_utime + ru.ru_stime
        except ChildProcessError:
            pid = self.pid
        if pid == 0:
            if time.time() > self.deadline:
                try:
                    os.kill(self.pid, signal.SIGKILL)
                    os.waitpid(self.pid, 0)
                except OSError:
                    pass
                return self._finish({"ok": False, "failed": [], "err": "deadline"})
            return None
        while True:
            chunk = os.read(self.fd, 65536)
            if not chunk:
                break
            self.buf += chunk
        try:
            return self._finish(json.loads(self.buf))
        except Exception:
            return self._finish({"ok": False, "failed": [], "err": "no result"})

    def _finish(self, res):
        res["cpu"] = getattr(self, "cpu", None)
        os.close(self.fd)
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)
        return res
