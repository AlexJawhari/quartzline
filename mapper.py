"""Turn a parsed patch into a conservative set of candidate tests (an int bitset over test numbers).

Built from the pinned commit only: per-line coverage per test (data/lines), classes touched per test
(data/classes), files opened per test (data/opens) and static AST/import analysis of the pristine source.
"""
import ast
import json
import marshal
import os
import bisect
import re
import zlib

import patchlib


def _load(path):
    with open(path, "rb") as f:
        return marshal.loads(zlib.decompress(f.read()))


def _bits(b):
    return int.from_bytes(b, "little")


class Mapper:
    def __init__(self, data_dir, src_root, tests):
        self.data, self.src = data_dir, src_root
        self.tests = tests
        self.all_bits = (1 << len(tests)) - 1
        self.classes = {k: _bits(v) for k, v in _load(f"{data_dir}/classes.z").items()}
        self.opens = {k: _bits(v) for k, v in _load(f"{data_dir}/opens.z").items()}
        self.imp = {k: set(v) for k, v in _load(f"{data_dir}/imp.z").items()}
        with open(f"{data_dir}/modtouch.json") as f:
            self.modtouch = json.load(f)  # class -> modules whose import instantiates it
        if os.environ.get("BLAST_NO_MODTOUCH"):  # evaluation only: simulate the missing map
            self.modtouch = {}
        with open(f"{data_dir}/closure.json") as f:
            self.closure = json.load(f)  # sqlglot file -> test files whose import loads it
        self._lines, self._any, self._ast, self._text, self._ilines = {}, {}, {}, {}, {}
        self.by_file = {}
        for t in tests:
            self.by_file[t["file"]] = self.by_file.get(t["file"], 0) | (1 << t["number"])
        self._test_srcs = None

    # ---- data access
    def lines(self, rel):
        d = self._lines.get(rel)
        if d is None:
            p = f"{self.data}/lines/{rel.replace('/', '__')}.z"
            d = {l: _bits(v) for l, v in _load(p).items()} if os.path.exists(p) else {}
            self._lines[rel] = d
            a = 0
            for v in d.values():
                a |= v
            for v in self.ilines(rel).values():
                a |= v
            self._any[rel] = a
        return d

    def ilines(self, rel):
        """Module/class-body lines executed inside a test (lazy imports)."""
        d = self._ilines.get(rel)
        if d is None:
            p = f"{self.data}/ilines/{rel.replace('/', '__')}.z"
            d = {l: _bits(v) for l, v in _load(p).items()} if os.path.exists(p) else {}
            self._ilines[rel] = d
        return d

    def any_line(self, rel):
        self.lines(rel)
        return self._any[rel]

    def text(self, rel):
        if rel not in self._text:
            try:
                with open(f"{self.src}/{rel}", encoding="utf-8") as f:
                    self._text[rel] = f.read()
            except OSError:
                self._text[rel] = None
        return self._text[rel]

    def tree(self, rel):
        if rel not in self._ast:
            try:
                self._ast[rel] = ast.parse(self.text(rel))
            except Exception:
                self._ast[rel] = None
        return self._ast[rel]

    # ---- helpers
    @staticmethod
    def modname(rel):
        m = rel[:-3].replace("/", ".")
        return m[: -len(".__init__")] if m.endswith(".__init__") else m

    def union_lines(self, rel, lo, hi, imp=False):
        d, di = self.lines(rel), self.ilines(rel)
        r = 0
        for l in range(lo, hi + 1):
            r |= d.get(l, 0)
            if imp:
                r |= di.get(l, 0)
        return r

    def files_mentioning(self, word, exclude=None):
        pat = re.compile(r"\b%s\b" % re.escape(word))
        r = 0
        for rel in self.sqlglot_files():
            if pat.search(self.text(rel) or ""):
                r |= self.any_line(rel)
        return r

    def sqlglot_files(self):
        if not hasattr(self, "_sf"):
            self._sf = []
            for dp, _, fs in os.walk(f"{self.src}/sqlglot"):
                for f in fs:
                    if f.endswith(".py"):
                        self._sf.append(os.path.relpath(f"{dp}/{f}", self.src))
        return self._sf

    def mentions(self, name):
        """True when the word `name` appears in any sqlglot or test source."""
        pat = re.compile(r"\b%s\b" % re.escape(name))
        if not hasattr(self, "_all_text"):
            self._all_text = [self.text(f) or "" for f in self.sqlglot_files()]
            for dp, _, fs in os.walk(f"{self.src}/tests"):
                for f in fs:
                    if f.endswith(".py"):
                        with open(f"{dp}/{f}", encoding="utf-8", errors="replace") as h:
                            self._all_text.append(h.read())
        return any(name in t and pat.search(t) for t in self._all_text)

    def importer_files(self, rel):
        """Source files that import module `rel` (or names from it). Memoised: texts are the pristine tree."""
        memo = self.__dict__.setdefault("_impf", {})
        if rel not in memo:
            memo[rel] = self._importer_files(rel)
        return list(memo[rel])

    def _importer_files(self, rel):
        mod = self.modname(rel)
        leaf = mod.rsplit(".", 1)[-1]
        pkg = mod.rsplit(".", 1)[0]
        direct = re.compile(r"(from\s+%s\s+import|import\s+%s\b)" % (re.escape(mod), re.escape(mod)))
        sub = re.compile(r"from\s+%s\s+import\s+(\([^)]*\)|[^\n]*)" % re.escape(pkg))
        leafw = re.compile(r"\b%s\b" % re.escape(leaf))

        class _P:
            @staticmethod
            def search(text):
                return direct.search(text) or any(leafw.search(g) for g in sub.findall(text))

        pat = _P
        return [f for f in self.sqlglot_files() if f != rel and leaf in (self.text(f) or "") and pat.search(self.text(f) or "")]

    def importers(self, rel):
        """Bits of tests that executed anything in a sqlglot file that imports module `rel`."""
        memo = self.__dict__.setdefault("_impb", {})
        if rel not in memo:
            r = 0
            for f in self.importer_files(rel):
                r |= self.any_line(f)
            memo[rel] = r
        return memo[rel]

    # ---- scope analysis of one old-file line
    def _chain(self, tree, line):
        """Ancestors (outermost first) of statements containing `line`."""
        chain, body = [], tree.body
        while True:
            for n in body:
                lo = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
                if lo <= line <= n.end_lineno:
                    chain.append(n)
                    break
            else:
                return chain
            body = self._children(chain[-1])
            if not body:
                return chain

    @staticmethod
    def _children(n):
        out = []
        for f in ("body", "orelse", "finalbody"):
            out += getattr(n, f, []) or []
        for h in getattr(n, "handlers", []) or []:
            out += h.body
        for c in getattr(n, "cases", []) or []:
            out += c.body
        return out

    def _own_range(self, n):
        """Lines of statement n not inside any child statement."""
        lo = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        kids = self._children(n)
        if not kids:
            return set(range(lo, n.end_lineno + 1))
        inner = set()
        for k in kids:
            inner |= set(range(min([k.lineno] + [d.lineno for d in getattr(k, "decorator_list", [])]), k.end_lineno + 1))
        return set(range(lo, n.end_lineno + 1)) - inner

    def line_bits(self, rel, line, refine=True, span=None):
        """Candidate bits for a change at old-file `line` of sqlglot source file rel."""
        tree = self.tree(rel)
        if tree is None:
            return self.any_line(rel) | self.importers(rel)
        chain = self._chain(tree, line)
        if not chain:
            return self.lines(rel).get(line, 0)  # blank/comment-only region
        inner = chain[-1]
        funcs = [n for n in chain if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [n for n in chain if isinstance(n, ast.ClassDef)]
        own = self._own_range(inner)
        r = 0
        mod = self.modname(rel)
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)) and line in own:
            self._flag = True  # decorators / defaults run at import
            # signature / decorators / defaults: every caller of the function
            r |= self.union_lines(rel, inner.lineno, inner.end_lineno)
            r |= self.importers(rel) if not funcs[:-1] and not classes else 0
            if not funcs[:-1]:
                r |= self._class_bits(mod, classes)
            return r
        if not funcs and span and not isinstance(inner, ast.ClassDef):
            lam = self._lambda_at(inner, line, span)
            if lam is not None:
                return self.union_lines(rel, lam.lineno, lam.end_lineno)
        r |= self.union_lines(rel, min(own), max(own), imp=not funcs) if own else 0
        if not funcs:
            self._flag = True  # module / class body statement: runs at import
        if funcs:
            if own and any(l in self.imp.get(rel, ()) for l in own):
                self._flag = True
                # also executed while importing (metaclass / helper): effect reaches import-time state
                r |= self.any_line(rel) | self.importers(rel)
            return r
        # import-time statement (module or class level)
        names = self._assigned(inner)
        if classes:
            cb = self._class_bits(mod, classes)
            mc = self._mod_consumers(mod, classes)
            if isinstance(inner, ast.ClassDef):
                return r | cb | mc
            ub = self._usage_bits(rel, names, own) if names and refine else None
            if ub is None:
                return r | cb | mc
            return (cb & ub) | mc
        ub = self._usage_bits(rel, names, own) if names and refine else None
        if ub is None:
            return r | self.any_line(rel) | self.importers(rel)
        return r | ub

    @staticmethod
    def _lambda_at(stmt, line, span):
        """Innermost Lambda of stmt that contains the whole changed span on `line`, else None."""
        best = None
        for n in ast.walk(stmt):
            if isinstance(n, ast.Lambda):
                a, b = (n.lineno, n.col_offset), (n.end_lineno, n.end_col_offset)
                if a <= (line, span[0]) and (line, span[1]) <= b:
                    # the lambda's own parameters/defaults stay with the enclosing statement
                    if (line, span[0]) < (n.body.lineno, n.body.col_offset):
                        continue
                    best = n
        return best

    @staticmethod
    def _assigned(stmt):
        names = set()
        if isinstance(stmt, ast.Assign):
            tg = stmt.targets
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
            tg = [stmt.target]
        else:
            return names
        for t in tg:
            if isinstance(t, ast.Name):
                names.add(t.id)
            else:
                return set()  # tuple / attribute targets: do not refine
        return names

    def _usage_bits(self, rel, names, own):
        """Union of candidate bits over every other source line that mentions one of names.
        None when the name is too generic or a usage cannot be tied to tests."""
        pat = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(n) for n in names))
        defn = re.compile(r"^\s*(?:%s)\s*(?::[^=\n]+)?=(?!=)(.*)$" % "|".join(re.escape(n) for n in names))
        r = 0
        for f in self.sqlglot_files():
            txt = self.text(f) or ""
            if not any(n in txt for n in names):
                continue
            starts = self.__dict__.setdefault("_starts", {})
            if f not in starts:
                starts[f] = [0] + [m.end() for m in re.finditer("\n", txt)]
            hits = {bisect.bisect_right(starts[f], m.start()) for m in pat.finditer(txt)}
            if not hits:
                continue
            lines = txt.split("\n")
            for i in sorted(hits):
                ln = lines[i - 1]
                if f == rel and i in own:
                    continue
                if not ln.lstrip().startswith("#"):
                    m = defn.match(ln)
                    if m and not pat.search(m.group(1)):
                        continue  # an override / redefinition, not a use
                    r |= self.line_bits(f, i, refine=False)
        return r

    def _mod_consumers(self, mod, classes):
        """Class-body code elsewhere builds instances of this class at import (e.g. a parsed SQL
        template): every test that loads that module sees the changed class definition."""
        if not classes:
            return 0
        q = mod + "." + ".".join(c.name for c in classes)
        r = 0
        for rel in self.modtouch.get(q, ()):
            r |= self.any_line(rel) | self.importers(rel)
            for f in self.closure.get(rel, ()):
                r |= self.by_file.get(f, 0)
        return r

    def _class_bits(self, mod, classes):
        if not classes:
            return 0
        q = mod + "." + ".".join(c.name for c in classes)
        r = self.classes.get(q, 0)
        return r

    # ---- patch level
    def candidates(self, fds, orig):
        """orig(path)->text|None. Returns (bits, notes)."""
        bits, notes, self.assumed = 0, [], 0
        self.import_files, self._flag = set(), False
        prev = None
        for fd in fds:
            if self._flag and prev:
                self.import_files.add(prev)
            self._flag, prev = False, (fd.old or fd.path)
            path = fd.old or fd.path
            if path.endswith(".py") and path.startswith("sqlglot/"):
                old = orig(path)
                if old is None:
                    continue
                new = patchlib.apply(fd, old)
                if new is not None and not self._compiles(new, path):
                    bits |= self.any_line(path) | self.importers(path)  # import error: everything that loads it
                    continue
                if new is not None and patchlib.tokens_equal(old, new):
                    notes.append(f"{path}: tokens equal")
                    continue
                if new is None:
                    bits |= self.any_line(path) | self.importers(path)
                    continue
                removed, ins = patchlib.changed_old_lines(fd)
                spans = patchlib.changed_spans(fd)
                for l in sorted(removed):
                    bits |= self.line_bits(path, l, span=spans.get(l))
                for a, b in ins:
                    for l in self._stmt_neighbors(path, a, b):
                        bits |= self.line_bits(path, l)
            elif path.startswith("tests/") and path.endswith(".py"):
                bits |= self._test_file_bits(fd, orig)
                self._flag = False
            elif path.startswith("tests/"):
                fb = self._data_file_bits(path)
                if self._real_edit(fd):
                    self.assumed |= fb  # a corrupted fixture line fails its reader: print, don't run
                else:
                    bits |= fb
            elif path.startswith("sqlglot"):
                bits |= self._data_file_bits(path)  # package data
            else:
                bits |= self._data_file_bits(path)
        if self._flag and prev:
            self.import_files.add(prev)
        return bits & self.all_bits, notes

    def import_fail_bits(self, rel):
        """Tests that fail when module `rel` cannot be imported."""
        if rel not in self.closure:
            return self.all_bits if rel in self.imp else self.any_line(rel) | self.importers(rel)
        r = self.any_line(rel) | self.importers(rel)
        for f in self.closure[rel]:
            r |= self.by_file.get(f, 0)
        return r

    @staticmethod
    def _real_edit(fd):
        for _, body in fd.hunks:
            for op, t in body:
                t = t.strip()
                if op != " " and t and not t.startswith(("--", "#")):
                    return True
        return False

    @staticmethod
    def _compiles(text, path):
        try:
            compile(text, path, "exec")
            return True
        except Exception:
            return False

    def _stmt_neighbors(self, rel, a, b):
        txt = (self.text(rel) or "").split("\n")
        out = []
        i = a
        while i >= 1 and (not txt[i - 1].strip() or txt[i - 1].lstrip().startswith("#")):
            i -= 1
        if i >= 1:
            out.append(i)
        j = b
        while j <= len(txt) and (not txt[j - 1].strip() or txt[j - 1].lstrip().startswith("#")):
            j += 1
        if j <= len(txt):
            out.append(j)
        return out

    def _data_file_bits(self, path):
        r = self.opens.get(path, 0)
        if r:
            return r
        return self._mentions_in_tests(os.path.basename(path))  # no recorded reader: grep the tests

    def _test_sources(self):
        if self._test_srcs is None:
            self._test_srcs = {}
            for rel in self.by_file:
                self._test_srcs[rel] = self.text(rel) or ""
        return self._test_srcs

    def _mentions_in_tests(self, word):
        r = 0
        for rel, s in self._test_sources().items():
            if word in s:
                r |= self.by_file[rel]
        return r

    def _test_file_bits(self, fd, orig):
        path = fd.old or fd.path
        r = 0
        old = orig(path)
        tree = None
        try:
            tree = ast.parse(old) if old else None
        except Exception:
            pass
        if fd.new is None or tree is None or not self._compiles(patchlib.apply(fd, old), path):
            return self.by_file.get(path, 0) | self._test_importers(path)
        removed, ins = patchlib.changed_old_lines(fd)
        lines = set(removed)
        for a, b in ins:
            lines |= {a, b}
        by_name = {}
        for t in self.tests:
            if t["file"] == path:
                _, c, n = t["id"].split("::")
                by_name[(c, n)] = 1 << t["number"]
        for l in lines:
            hit = None
            for n in tree.body:
                if isinstance(n, ast.ClassDef) and n.lineno <= l <= n.end_lineno:
                    for m in n.body:
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                                min([m.lineno] + [d.lineno for d in m.decorator_list]) <= l <= m.end_lineno and \
                                (n.name, m.name) in by_name and l > m.lineno:
                            hit = by_name[(n.name, m.name)]
            if hit is None:
                r |= self.by_file.get(path, 0) | self._test_importers(path)
            else:
                r |= hit
        return r

    def _test_importers(self, path):
        mod = path[:-3].replace("/", ".")
        leaf = mod.rsplit(".", 1)[-1]
        r = 0
        for rel, s in self._test_sources().items():
            if rel != path and (mod in s or re.search(r"\bimport\s+[^\n]*\b%s\b" % re.escape(leaf), s)
                                or re.search(r"from\s+(?:tests\.)?%s\s+import" % re.escape(leaf), s)):
                r |= self.by_file[rel]
        return r
