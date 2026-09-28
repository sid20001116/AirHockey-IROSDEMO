#!/usr/bin/env python3
"""Browser GUI for ``mini_platform.hw.run_hw_match``.

Run it, open the page, set the flags you want, press Start::

    python -m mini_platform.hw.match_gui        # -> http://127.0.0.1:8765

The flag list is read out of ``run_hw_match.py`` itself at startup -- its ``add_argument``
calls are parsed with ``ast``, so nothing is imported and the GUI cannot drift from the
CLI. Add a flag there and it appears here, with its help text, default and choices.

Stop sends **SIGINT**, the signal Ctrl-C sends, because ``run_hw_match`` catches
``KeyboardInterrupt`` to save the ``--log`` telemetry and to run ``arm.catch_and_close()``
in its ``finally``. A hard kill skips both and leaves the arm uncaught, so force-kill is a
separate button that only appears once a graceful stop has been given time to finish.

Stdlib only -- no pip install, no build step.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import json
import os
import pty
import queue
import re
import shlex
import signal
import struct
import subprocess
import termios
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_HERE = Path(__file__).resolve()
CLI_PATH = _HERE.with_name("run_hw_match.py")
CLI_MODULE = "mini_platform.hw.run_hw_match"
REPO_ROOT = _HERE.parents[3]                      # .../metal-python-ros2
STATE_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "metal_match_gui"

MAX_LINES = 8000                                  # log ring buffer
GRACE_SECONDS = 20                                # before the force-kill button appears


# --------------------------------------------------------------------------------------
# CLI introspection
# --------------------------------------------------------------------------------------

def _lit(node):
    """Literal value of an AST node, or its source text when it is an expression."""
    try:
        return ast.literal_eval(node), True
    except Exception:
        return ast.unparse(node), False


# The page's sections, in order, each listing its flags by name in the order they should be
# shown. This order also drives the order flags are emitted on the command line, so a
# generated line reads the way a hand-written one does.
#
# Assignment is by exact name on purpose: a flag added to run_hw_match later lands in
# "Other" and stays visible, rather than being silently swallowed by a prefix rule.
GROUPS = [
    ("Run",
     "Which stage to run, and where the telemetry goes.",
     ["stage", "task", "duration", "log", "view", "mock"]),

    ("Calibration files",
     "The fits this run loads: camera extrinsics, table frame, puck colour.",
     ["calib", "base-frame", "puck-profile", "ref-world-xy", "out-calib", "out"]),

    ("Camera",
     "Capture settings. A puck profile is only valid at the exposure it was tuned at.",
     ["camera-index", "cam-fps", "cam-exposure", "cam-auto-exposure", "cam-auto-priority",
      "no-table-mask"]),

    ("Puck tracking (EKF)",
     "Where the puck is, and where it is going next.",
     ["ekf-params", "sigma-px", "max-puck-speed", "vel-lpf-hz", "meas-age-lead",
      "meas-lag-comp", "intercept-max-jump", "intercept-ema-alpha", "lead-time",
      "max-prediction-time"]),

    ("Mallet & table height",
     "How the mallet is held, and how low it rides.",
     ["flat-mallet", "ee-height", "ee-height-npz", "contact-offset", "init-lift",
      "gripper-mm"]),

    ("Joint speed & acceleration",
     "How fast each joint may move. One value for all, or six: J1,J2,J3,J4,J5,J6.",
     ["max-joint-vel", "max-joint-accel", "velocity-ratio", "home-velocity-ratio", "slow"]),

    ("Homing & RT stream",
     "How commands reach the arm, and what it does before the match starts.",
     ["rt-home", "rt-home-time", "rt-settle", "stream-hz", "stream-limit", "stream-lead-ms",
      "stream-couple-spin", "stream-include-gripper", "resync-cmd", "resync-thresh",
      "wall-clock", "health-hz"]),

    ("Home & park pose",
     "Where the arm sits when it has nothing to chase.",
     ["home", "home-x", "home-y", "home-goal-dist"]),

    ("Goalie",
     "--goalie switches this whole group on; each knob is also settable on its own.",
     ["goalie", "ekf-own-goal", "gate-unled", "fresh-horizon", "rendezvous-vel",
      "rendezvous-from-measured"]),

    ("Defend",
     "Where the mallet plants to block a shot, and when it gives up on one.",
     ["defend-line-frac", "defend-width-frac", "defend-block", "defend-centre-deadband",
      "defend-abort-dist", "defend-give-up-ticks", "defend-hold-target", "defend-early-frac",
      "defend-early-cos"]),

    ("Clear the puck",
     "Return the puck up-table instead of tapping it sideways. The cock drives the mallet "
     "toward our own goal, so it is opt-in and not part of --goalie.",
     ["defend-clear", "defend-clear-speed", "defend-clear-standoff", "defend-clear-margin",
      "defend-clear-max-path"]),

    ("Ready guard",
     "While idle, wait on the puck-to-goal line instead of at a fixed park.",
     ["ready-guard", "ready-guard-radius", "ready-guard-drift", "ready-t-stop-min"]),

    ("Attack",
     "Striking a slow puck, and when to switch tactic.",
     ["smash-slow", "smash-slow-vel", "smash-slow-ticks", "smash-slow-max-x",
      "static-vel-threshold", "switch-tactics-min-steps"]),

    ("Planner",
     "How the path from here to the intercept is generated.",
     ["planner", "max-plan-steps", "workspace-clamp", "vpsto-vel-lim", "vpsto-acc-lim",
      "vpsto-smooth", "vpsto-max-iter"]),

    ("Keep-outs",
     "Regions the arm must not enter.",
     ["min-goal-dist", "goal-keepout-hard-margin", "no-goal-keepout", "base-keepout",
      "base-keepout-hard-margin"]),

    ("Arm hardware",
     "SDK and robot setup. Rarely changed once the rig is up.",
     ["can-id", "urdf", "arm-end-type"]),

    ("Track test",
     "Only read by --stage track-test.",
     ["track-joint", "track-amp", "track-freq"]),
]

OTHER_BLURB = "Not yet placed in a section -- newly added to run_hw_match.py."

#: flag name -> (section title, position in the whole ordered list)
_PLACEMENT = {name: (title, i)
              for i, (title, name) in enumerate(
                  (t, n) for t, _, names in GROUPS for n in names)}

# Existence of these is checked before a run: a typo here costs a rig session.
INPUT_PATHS = {"calib", "base-frame", "puck-profile", "ee-height-npz", "ekf-params", "urdf"}
OUTPUT_PATHS = {"log", "out", "out-calib"}


def _group_for(name: str) -> str:
    return _PLACEMENT.get(name, ("Other", 0))[0]


def parse_goalie_defaults(tree: ast.Module) -> dict:
    """What ``--goalie`` switches on, read from ``GOALIE_DEFAULTS`` so it cannot go stale.

    Without this the GUI would warn about ``--lead-time`` needing ``--gate-unled`` even
    though ``--goalie`` already turns that on.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "GOALIE_DEFAULTS" for t in node.targets):
            try:
                raw = ast.literal_eval(node.value)
            except Exception:
                return {}
            return {k.replace("_", "-"):
                    ("on" if v is True else "off" if v is False else str(v))
                    for k, v in raw.items()}
    return {}


def parse_cli_spec(path: Path) -> list[dict]:
    """Every ``--flag`` ``run_hw_match.py`` accepts, as plain dicts for the browser."""
    tree = ast.parse(path.read_text())
    goalie = parse_goalie_defaults(tree)
    out: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        names = [_lit(a)[0] for a in node.args]
        if not names or not isinstance(names[0], str) or not names[0].startswith("--"):
            continue
        kw = {k.arg: _lit(k.value) for k in node.keywords}
        name = names[0][2:]

        action = kw.get("action", ("store", True))[0]
        default, default_is_literal = kw.get("default", (None, True))
        choices = kw.get("choices", (None, True))[0]
        nargs = kw.get("nargs", (None, True))[0]
        type_ = kw.get("type", ("str", False))[0]
        help_ = kw.get("help", ("", True))[0]
        if not isinstance(help_, str):
            help_ = ""

        if action == "store_true":
            kind = "flag"
        elif "BooleanOptionalAction" in str(action):
            kind = "tri"
        elif choices:
            kind = "choice"
        elif nargs == 2:
            kind = "pair"
        elif type_ in ("float", "int"):
            kind = "number"
        else:
            kind = "text"

        if default_is_literal:
            shown = "" if default is None else ("on" if default is True else
                                                "off" if default is False else str(default))
        else:
            shown = str(default)                  # e.g. DEFAULT_URDF -- show the name

        out.append({
            "name": name,
            "kind": kind,
            "help": " ".join(help_.replace("%%", "%").split()),
            "default": shown,
            "choices": [str(c) for c in choices] if choices else None,
            "required": bool(kw.get("required", (False, True))[0]),
            "group": _group_for(name),
            "is_input_path": name in INPUT_PATHS,
            "is_output_path": name in OUTPUT_PATHS,
            "goalie_default": goalie.get(name),
        })

    # Section order, then the order listed within the section. Anything unplaced keeps its
    # source order and falls to the end, under "Other".
    out.sort(key=lambda f: _PLACEMENT.get(f["name"], ("", len(_PLACEMENT)))[1])
    return out


#: Shown in a "Pinned" card above the sections, so the knobs that change between runs are at
#: hand without opening anything. The star on any row adds or removes one.
DEFAULT_PINNED = ["rt-home", "rt-home-time", "max-joint-vel", "max-joint-accel",
                  "cam-exposure", "log"]

# The command the user actually runs today, as the starting preset.
BUILTIN_PRESETS = {
    "Goalie + clear (current rig)": {
        "stage": "play",
        "calib": "hw_calib_teleop.npz",
        "base-frame": "base_frame_teleop.npz",
        "puck-profile": "puck_profile.json",
        "ee-height-npz": "ee_height.npz",
        "camera-index": "0",
        "cam-fps": "30",
        "cam-auto-exposure": "1",
        "cam-exposure": "150",
        "cam-auto-priority": "0",
        "ekf-params": "mini_table_ekf.npz",
        "sigma-px": "0.56",
        "max-puck-speed": "30.0",
        "flat-mallet": True,
        "rt-home": True,
        "rt-home-time": "5",
        "max-joint-vel": "1.8,1.4,1.4,1.4,1.4,1.4",
        "max-joint-accel": "70,140,65,130,40,40",
        "goalie": True,
        "log": "clear_early.npz",
        "lead-time": "0.05",
        "max-plan-steps": "3",
        "defend-width-frac": "1.3",
        "defend-clear": True,
        "defend-early-frac": "0.6",
    },
}


def build_argv(values: dict, spec: list[dict]) -> list[str]:
    """Flag values -> argv, emitting only what differs from the CLI's own defaults."""
    argv = ["python", "-u", "-m", CLI_MODULE]
    for s in spec:
        name, kind = s["name"], s["kind"]
        v = values.get(name, "")
        if kind == "flag":
            if v is True or v == "on":
                argv.append(f"--{name}")
        elif kind == "tri":
            if v == "on":
                argv.append(f"--{name}")
            elif v == "off":
                argv.append(f"--no-{name}")
        elif kind == "pair":
            parts = str(v).replace(",", " ").split()
            if len(parts) == 2:
                argv += [f"--{name}", parts[0], parts[1]]
        else:
            text = str(v).strip()
            if text:
                argv += [f"--{name}", text]
            elif s["required"]:
                argv += [f"--{name}", str(s["choices"][0] if s["choices"] else "")]
    return argv


TUNE_MODULE = "mini_platform.hw.tune_puck"
#: Camera flags tune_puck shares with run_hw_match, in the order its own docstring lists them.
_TUNE_CAM_FLAGS = ("camera-index", "cam-fps", "cam-exposure", "cam-auto-exposure",
                   "cam-auto-priority")
#: tune_puck types these as int where run_hw_match takes a float, so "150.0" would be rejected.
_TUNE_INT_FLAGS = {"camera-index", "cam-exposure", "cam-auto-exposure", "cam-auto-priority"}


def build_tune_argv(values: dict, resolution: str = "") -> list[str]:
    """``tune_puck``, at the camera settings the play form is set to.

    Its docstring is emphatic about this: ``sat_min``/``val_min`` are fixed thresholds, so a
    profile fitted at one exposure mis-segments at another -- and ``--stage play`` warns when
    it is handed a profile fitted somewhere else. Taking the camera flags and the output path
    from the same form that runs the match is what keeps the two in step.
    """
    argv = ["python", "-u", "-m", TUNE_MODULE]
    for name in _TUNE_CAM_FLAGS:
        v = str(values.get(name, "") or "").strip()
        if not v:
            continue
        if name in _TUNE_INT_FLAGS:
            try:
                v = str(int(float(v)))
            except ValueError:
                pass
        argv += [f"--{name}", v]
    argv += ["--resolution", resolution.strip() or "640x480"]
    argv += ["--out", str(values.get("puck-profile", "") or "").strip() or "puck_profile.json"]
    return argv


def preview_warnings(values: dict, spec: list[dict], cwd: str) -> list[str]:
    """Cheap pre-flight checks. Everything here costs a rig run if it goes unnoticed."""
    warns: list[str] = []
    by_name = {s["name"]: s for s in spec}

    def resolve(p: str) -> Path:
        q = Path(os.path.expanduser(p))
        return q if q.is_absolute() else Path(cwd) / q

    for name, s in by_name.items():
        v = str(values.get(name, "") or "").strip()
        if not v:
            continue
        if s["is_input_path"] and not resolve(v).exists():
            warns.append(f"--{name}: no such file — {v}")
        if s["is_output_path"] and resolve(v).exists():
            warns.append(f"--{name}: {v} already exists and will be overwritten")
        if s["kind"] == "pair" and len(v.replace(",", " ").split()) != 2:
            warns.append(f"--{name} wants two numbers; {v!r} will be dropped")

    # Straight out of --gate-unled's own help text: a lead above ~20 ms applied to the gated
    # intercept disables DEFEND entirely. --goalie switches --gate-unled on by itself, and an
    # explicit --no-gate-unled survives it (apply_goalie_defaults only fills in None), so the
    # check has to resolve the flag the same way the program does.
    try:
        lead = float(str(values.get("lead-time", "") or 0) or 0)
    except ValueError:
        lead = 0.0
    gate = values.get("gate-unled", "")
    gated_unled = (gate == "on") or (gate == "" and values.get("goalie") is True
                                     and by_name.get("gate-unled", {}).get("goalie_default") == "on")
    if lead > 0.02 and not gated_unled:
        warns.append(
            f"--lead-time {lead:g} without --gate-unled: leads above ~20 ms drag the gated "
            "intercept below the defend line and disable DEFEND entirely (see its --help)")
    return warns


# --------------------------------------------------------------------------------------
# Process runner
# --------------------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-_]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(text: str) -> str:
    return _CTRL.sub("", _ANSI.sub("", text))


class Runner:
    """Owns at most one ``run_hw_match`` process and fans its output out over SSE."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.master_fd: int | None = None
        self.subs: list[queue.Queue] = []
        self.lines: list[str] = []
        self.base = 0                  # absolute index of self.lines[0]
        self._open_line = False        # last line has no newline yet
        self._buf = ""
        self.status = "idle"           # idle | running | stopping | exited
        self.cmdline = ""
        self.started_at: float | None = None
        self.stopping_since: float | None = None
        self.exit_code: int | None = None

    # -- subscribers ---------------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=4000)
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def _broadcast(self, event: dict) -> None:
        with self.lock:
            subs = list(self.subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def state(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "cmdline": self.cmdline,
                "started_at": self.started_at,
                "exit_code": self.exit_code,
                "can_force": bool(self.stopping_since
                                  and time.time() - self.stopping_since > GRACE_SECONDS),
            }

    def snapshot(self) -> dict:
        with self.lock:
            return {"type": "snapshot", "base": self.base,
                    "lines": list(self.lines), "state": self.state()}

    # -- log assembly --------------------------------------------------------------
    @staticmethod
    def _visible(chunk: str) -> str:
        """What a terminal would show for one line: only the last carriage-return segment.

        The trailing ``\\r`` is dropped first -- a pty translates every ``\\n`` the child
        writes into ``\\r\\n``, so without that every line would come out empty.
        """
        return _clean(chunk.rstrip("\r").split("\r")[-1])

    def _emit(self, text: str, final: bool, force_new: bool = False) -> None:
        """Append a line, or rewrite the last one when it was left open by a bare ``\\r``."""
        with self.lock:
            if self._open_line and self.lines and not force_new:
                self.lines[-1] = text
            else:
                self.lines.append(text)
                if len(self.lines) > MAX_LINES:
                    drop = len(self.lines) - MAX_LINES
                    del self.lines[:drop]
                    self.base += drop
            self._open_line = not final
            idx = self.base + len(self.lines) - 1
            self._broadcast({"type": "line", "i": idx, "t": text})

    def _feed(self, data: str) -> None:
        # Held across the whole assembly: log_note() runs on an HTTP thread and would
        # otherwise interleave with this one and overwrite a half-built line.
        with self.lock:
            parts = (self._buf + data).split("\n")
            self._buf = parts.pop()
            for p in parts:
                self._emit(self._visible(p), final=True)
            if self._buf:
                self._emit(self._visible(self._buf), final=False)

    def log_note(self, text: str) -> None:
        """A line from the GUI itself. Never overwrites the program's own output."""
        with self.lock:
            self._buf = ""
            self._emit(text, final=True, force_new=True)

    # -- lifecycle -----------------------------------------------------------------
    def start(self, argv: list[str], cwd: str, prelude: str) -> None:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("a run is already in flight")
            self.lines, self.base, self._buf, self._open_line = [], 0, "", False
            self.exit_code, self.stopping_since = None, None

        shown = " ".join(shlex.quote(a) for a in argv)
        if prelude.strip():
            inner = f"cd {shlex.quote(cwd)} && exec {shown}"
            cmd = ["bash", "-lc", f"{prelude.strip()} && {inner}"]
        else:
            cmd = argv

        # A pty, not a pipe: the SDK's C++ side and Python both line-buffer only when
        # stdout looks like a terminal. Through a pipe the log would arrive in 4 KB lumps.
        master, slave = pty.openpty()
        try:
            fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 200, 0, 0))
            attrs = termios.tcgetattr(slave)
            attrs[1] &= ~termios.ONLCR          # don't rewrite the child's \n as \r\n
            attrs[3] &= ~termios.ECHO           # don't echo what we send to stdin
            termios.tcsetattr(slave, termios.TCSANOW, attrs)
        except (OSError, termios.error):
            pass
        env = dict(os.environ, PYTHONUNBUFFERED="1", TERM="xterm-256color")
        proc = subprocess.Popen(cmd, cwd=cwd, stdin=slave, stdout=slave, stderr=slave,
                                start_new_session=True, close_fds=True, env=env)
        os.close(slave)

        with self.lock:
            self.proc, self.master_fd = proc, master
            self.status, self.started_at = "running", time.time()
            self.cmdline = shown
        self.log_note(f"$ {shown}")
        self._broadcast({"type": "state", "state": self.state()})
        threading.Thread(target=self._pump, args=(master, proc), daemon=True).start()

    def _pump(self, fd: int, proc: subprocess.Popen) -> None:
        try:
            while True:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                self._feed(chunk.decode("utf-8", "replace"))
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            code = proc.wait()
            if self._buf:
                self._feed("\n")                # flush a last line with no newline on it
            with self.lock:
                self.proc, self.master_fd = None, None
                self.status, self.exit_code = "exited", code
                self.stopping_since, self.started_at = None, None
            self.log_note(f"— exited with code {code} —")
            self._broadcast({"type": "state", "state": self.state()})

    def send_stdin(self, text: str) -> bool:
        """Answer a prompt. The calibration stages (set-origin, table-height) use input()."""
        with self.lock:
            fd = self.master_fd
            if fd is None:
                return False
            os.write(fd, (text + "\n").encode())
            self._emit("> " + text, final=True, force_new=True)
            return True

    def stop(self, force: bool = False) -> str:
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return "nothing to stop"
            pgid = os.getpgid(proc.pid)
            if force:
                os.killpg(pgid, signal.SIGKILL)
                self.log_note("— SIGKILL sent; the arm was NOT handed to gravity "
                              "compensation —")
                return "killed"
            if self.status != "stopping":
                self.status = "stopping"
                self.stopping_since = time.time()
                os.killpg(pgid, signal.SIGINT)
                self.log_note("— SIGINT sent (same as Ctrl-C): saving telemetry and "
                              "catching the arm —")
                self._broadcast({"type": "state", "state": self.state()})
                return "stopping"
            return "already stopping"

    def shutdown(self) -> None:
        """Never leave the arm live because the GUI went away."""
        with self.lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except OSError:
            return
        for _ in range(int(GRACE_SECONDS * 10)):
            if proc.poll() is not None:
                return
            time.sleep(0.1)


RUNNER = Runner()


# --------------------------------------------------------------------------------------
# Presets / session persistence
# --------------------------------------------------------------------------------------

def _load_json(path: Path, fallback):
    try:
        return json.loads(path.read_text())
    except Exception:
        return fallback


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)          # atomic, and unique per thread so previews cannot collide


PRESETS_FILE = STATE_DIR / "presets.json"
SESSION_FILE = STATE_DIR / "session.json"


def all_presets() -> dict:
    return {**BUILTIN_PRESETS, **_load_json(PRESETS_FILE, {})}


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "match_gui"

    spec: list[dict] = []
    cwd: str = str(REPO_ROOT)
    prelude: str = ""

    def log_message(self, fmt, *args):            # quiet; the run's own log is the point
        pass

    # -- helpers -------------------------------------------------------------------
    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # -- routes --------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/spec":
            session = _load_json(SESSION_FILE, {})
            self._json({
                "flags": self.spec,
                "groups": ([{"title": t, "blurb": b} for t, b, _ in GROUPS]
                           + [{"title": "Other", "blurb": OTHER_BLURB}]),
                "presets": all_presets(),
                "builtin": list(BUILTIN_PRESETS),
                "cwd": session.get("cwd") or self.cwd,
                "prelude": session.get("prelude", self.prelude),
                "values": session.get("values") or next(iter(BUILTIN_PRESETS.values())),
                "preset": session.get("preset") or next(iter(BUILTIN_PRESETS)),
                "resolution": session.get("resolution") or "640x480",
                "pinned": session.get("pinned", DEFAULT_PINNED),
                "module": CLI_MODULE,
            })
        elif path == "/api/state":
            self._json(RUNNER.state())
        elif path == "/api/events":
            self._events()
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            body = self._body()
        except Exception as e:
            return self._json({"error": f"bad request: {e}"}, 400)

        if path == "/api/preview":
            values, cwd = body.get("values", {}), body.get("cwd") or self.cwd
            argv = build_argv(values, self.spec)
            # Only overwrite what was actually sent: "" is a legitimate prelude (inherit this
            # environment), so it must not be the fallback for a request that omits the key --
            # that would quietly drop the conda/ROS setup and run against the wrong python.
            session = _load_json(SESSION_FILE, {})
            session.update({"values": values, "cwd": cwd, "preset": body.get("preset", ""),
                            "resolution": body.get("resolution", ""),
                            "pinned": body.get("pinned", [])})
            if "prelude" in body:
                session["prelude"] = body["prelude"]
            _save_json(SESSION_FILE, session)
            return self._json({
                "cmd": " ".join(shlex.quote(a) for a in argv),
                "warnings": preview_warnings(values, self.spec, cwd),
                "tune_cmd": " ".join(shlex.quote(a) for a in
                                     build_tune_argv(values, body.get("resolution", ""))),
            })

        if path == "/api/start":
            values = body.get("values", {})
            cwd = body.get("cwd") or self.cwd
            if not Path(cwd).is_dir():
                return self._json({"error": f"working directory does not exist: {cwd}"}, 400)
            argv = (build_tune_argv(values, body.get("resolution", ""))
                    if body.get("tool") == "tune" else build_argv(values, self.spec))
            try:
                RUNNER.start(argv, cwd, body.get("prelude", ""))
            except Exception as e:
                return self._json({"error": str(e)}, 409)
            return self._json({"ok": True})

        if path == "/api/stop":
            return self._json({"ok": True, "result": RUNNER.stop(bool(body.get("force")))})

        if path == "/api/stdin":
            ok = RUNNER.send_stdin(str(body.get("text", "")))
            return self._json({"ok": ok} if ok else {"error": "nothing is running"},
                              200 if ok else 409)

        if path == "/api/presets":
            saved = _load_json(PRESETS_FILE, {})
            name = (body.get("name") or "").strip()
            if not name:
                return self._json({"error": "name required"}, 400)
            if body.get("delete"):
                saved.pop(name, None)
            else:
                saved[name] = body.get("values", {})
            _save_json(PRESETS_FILE, saved)
            return self._json({"presets": all_presets(), "builtin": list(BUILTIN_PRESETS)})

        return self._json({"error": "not found"}, 404)

    def _events(self):
        q = RUNNER.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self._push(RUNNER.snapshot())
            while True:
                try:
                    self._push(q.get(timeout=15))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            RUNNER.unsubscribe(q)

    def _push(self, event: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        self.wfile.flush()


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>run_hw_match</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2129; --line:#2a313c; --line2:#394150;
    --fg:#e6edf3; --muted:#8b949e; --accent:#4c8dff; --ok:#3fb950; --warn:#d29922;
    --bad:#f85149; --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,
       BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
  button{font:inherit;color:inherit;cursor:pointer;background:none;border:none}
  input,select{font:inherit;color:var(--fg);background:var(--bg);
       border:1px solid var(--line2);border-radius:6px;padding:5px 8px;min-width:0}
  input:focus,select:focus{outline:none;border-color:var(--accent)}
  input::placeholder{color:#5c6672}

  header{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:12px;
      flex-wrap:wrap;padding:10px 16px;background:rgba(13,17,23,.92);
      backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
  .brand{font-weight:650;letter-spacing:.2px}
  .brand span{color:var(--muted);font-weight:400;font-family:var(--mono);font-size:12px}
  .pill{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border-radius:999px;
      background:var(--panel2);border:1px solid var(--line);font-size:12.5px;
      font-family:var(--mono)}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
  .pill.running .dot{background:var(--ok);animation:pulse 1.4s infinite}
  .pill.stopping .dot{background:var(--warn);animation:pulse .7s infinite}
  .pill.exited .dot{background:var(--bad)}
  .pill.exited.clean .dot{background:var(--muted)}
  @keyframes pulse{50%{opacity:.25}}
  .spacer{flex:1}
  .btn{padding:7px 15px;border-radius:7px;border:1px solid var(--line2);
      background:var(--panel2);font-weight:550}
  .btn:hover:not(:disabled){border-color:var(--muted)}
  .btn:disabled{opacity:.38;cursor:not-allowed}
  .btn.go{background:var(--ok);border-color:var(--ok);color:#07210f}
  .btn.go:hover:not(:disabled){filter:brightness(1.1)}
  .btn.stop{background:#3d1d1c;border-color:#7a2b28;color:#ffb4b0}
  .btn.stop:hover:not(:disabled){background:#542422}
  .btn.kill{background:var(--bad);border-color:var(--bad);color:#2b0705}
  .btn.sm{padding:4px 10px;font-size:12.5px;font-weight:450}

  main{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);
      gap:14px;padding:14px;align-items:start}
  @media(max-width:1060px){main{grid-template-columns:1fr}}
  .col{display:flex;flex-direction:column;gap:12px;min-width:0}
  .col.right{position:sticky;top:62px}
  @media(max-width:1060px){.col.right{position:static}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      overflow:hidden}
  .card>h2{margin:0;padding:10px 14px;font-size:12px;font-weight:600;letter-spacing:.7px;
      text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--line);
      display:flex;align-items:center;gap:8px}
  .card>h2 .spacer{flex:1}
  .pad{padding:12px 14px}

  .toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .toolbar select{flex:1;min-width:160px}
  .toolbar input[type=search]{flex:1;min-width:150px}
  .check{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:12.5px;
      cursor:pointer;white-space:nowrap}

  details.group{border-top:1px solid var(--line)}
  details.group:first-of-type{border-top:none}
  details.group>summary{padding:9px 14px;cursor:pointer;font-weight:600;font-size:13px;
      display:flex;align-items:center;gap:8px;list-style:none}
  details.group>summary::-webkit-details-marker{display:none}
  details.group>summary::before{content:"\25B8";color:var(--muted);transition:transform .15s}
  details.group[open]>summary::before{transform:rotate(90deg)}
  .count{margin-left:auto;font-family:var(--mono);font-size:11.5px;color:var(--muted);
      background:var(--panel2);border:1px solid var(--line);border-radius:999px;padding:1px 8px}
  .count.on{color:var(--accent);border-color:#2a4a7a}
  .blurb{margin:-2px 14px 8px 28px;font-size:12px;color:var(--muted);max-width:62ch}

  .row{display:grid;grid-template-columns:minmax(150px,1.25fr) minmax(120px,1fr) auto auto;
      gap:8px;align-items:center;padding:5px 14px}
  .row:hover{background:#12171f}
  .row.set{box-shadow:inset 2px 0 0 var(--accent)}
  .row label{font-family:var(--mono);font-size:12.5px;color:#c9d4e0;cursor:help;
      overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .row.set label{color:var(--fg)}
  .row .req{color:var(--accent)}
  .row.bygoalie label::after{content:"--goalie";margin-left:6px;padding:0 5px;font-size:10px;
      color:var(--accent);border:1px solid #2a4a7a;border-radius:4px;vertical-align:1px}
  .clear,.pin{width:22px;height:22px;border-radius:5px;color:var(--muted);visibility:hidden;
      font-size:14px;line-height:1}
  .row.set .clear,.row:hover .clear{visibility:visible}
  .row:hover .pin,.row.pinned .pin{visibility:visible}
  .row.pinned .pin{color:var(--accent)}
  .clear:hover,.pin:hover{background:var(--panel2);color:var(--fg)}
  .subtle{font-weight:400;letter-spacing:0;text-transform:none;font-size:11.5px;
      color:var(--muted)}
  #pinned .row:first-child{padding-top:9px}
  #pinned .row:last-child{padding-bottom:9px}
  .row input,.row select{width:100%}

  .seg{display:flex;border:1px solid var(--line2);border-radius:6px;overflow:hidden}
  .seg button{flex:1;padding:4px 0;font-size:12px;color:var(--muted);text-align:center}
  .seg button+button{border-left:1px solid var(--line2)}
  .seg button.on{background:var(--accent);color:#06101f;font-weight:600}
  .seg button.on.neg{background:#5a626e;color:#0d1117}
  .sw{width:40px;height:22px;border-radius:999px;background:var(--line2);position:relative;
      transition:background .15s}
  .sw::after{content:"";position:absolute;top:2px;left:2px;width:18px;height:18px;
      border-radius:50%;background:var(--fg);transition:transform .15s}
  .sw.on{background:var(--accent)}
  .sw.on::after{transform:translateX(18px)}

  pre.cmd{margin:0;padding:12px 14px;font-family:var(--mono);font-size:12.5px;
      white-space:pre-wrap;word-break:break-all;color:#a9c9ff;max-height:190px;overflow:auto}
  pre.cmd.tune{padding:9px 11px;font-size:11.5px;background:var(--bg);
      border:1px solid var(--line);border-radius:7px;color:#8fb5e8}
  .prose{margin:0 0 10px;font-size:12.5px;color:var(--muted)}
  .prose code{font-family:var(--mono);font-size:11.5px;color:#c9d4e0}
  .prose.keys{margin:9px 0 11px}
  .prose.keys b{color:#c9d4e0;font-weight:600}
  .btn.wide{width:100%;text-align:center}
  .warn{display:flex;gap:9px;padding:9px 14px;border-top:1px solid var(--line);
      font-size:12.5px;color:#f0d08a;background:#2a2210}
  .warn::before{content:"\26A0";color:var(--warn)}
  .log{margin:0;padding:10px 14px;font-family:var(--mono);font-size:12px;line-height:1.55;
      white-space:pre-wrap;word-break:break-word;height:min(58vh,620px);overflow:auto;
      background:#0a0d12;border-radius:0 0 10px 10px}
  .log{border-radius:0}
  .log .note{color:var(--accent)}
  .log .empty{color:var(--muted)}
  .stdin{display:flex;align-items:center;gap:8px;padding:7px 14px;
      border-top:1px solid var(--line);font-family:var(--mono);color:var(--muted)}
  .stdin input{flex:1;border:none;background:none;padding:2px 0;font-size:12px}
  .stdin input:disabled{opacity:.5}
  .env{display:grid;grid-template-columns:auto 1fr;gap:8px 10px;align-items:center;
      font-size:12.5px;color:var(--muted)}
  .env input{font-family:var(--mono);font-size:12px}
  .toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);z-index:50;
      background:var(--panel2);border:1px solid var(--line2);border-radius:8px;
      padding:9px 16px;font-size:13px;box-shadow:0 8px 24px #0008;opacity:0;
      transition:opacity .2s,transform .2s;pointer-events:none}
  .toast.show{opacity:1;transform:translateX(-50%) translateY(-4px)}
  .toast.bad{border-color:var(--bad);color:#ffb4b0}
</style>
</head><body>

<header>
  <div class="brand">run_hw_match <span id="mod"></span></div>
  <div class="pill" id="status"><span class="dot"></span><span id="statusText">idle</span></div>
  <div class="spacer"></div>
  <button class="btn go" id="start">&#9654;&nbsp; Start</button>
  <button class="btn stop" id="stop" disabled>&#9632;&nbsp; Stop</button>
  <button class="btn kill" id="kill" hidden>Force kill</button>
</header>

<main>
  <div class="col">
    <div class="card" id="pinnedCard" hidden>
      <h2>Pinned<span class="spacer"></span>
        <span class="subtle">the knobs you change between runs</span></h2>
      <div id="pinned"></div>
    </div>

    <div class="card">
      <h2>Flags<span class="spacer"></span>
        <button class="btn sm" id="collapse">Collapse all</button>
        <button class="btn sm" id="reset">Reset all</button>
      </h2>
      <div class="pad toolbar">
        <select id="presetSel" title="Presets"></select>
        <button class="btn sm" id="presetSave">Save as&hellip;</button>
        <button class="btn sm" id="presetDel">Delete</button>
      </div>
      <div class="pad toolbar" style="padding-top:0">
        <input type="search" id="search" placeholder="Filter flags and help text&hellip;">
        <label class="check"><input type="checkbox" id="onlySet"> only set</label>
      </div>
      <div id="groups"></div>
    </div>

    <div class="card">
      <h2>Calibrate detection</h2>
      <div class="pad">
        <p class="prose">Runs <code>tune_puck</code> at the camera settings above and writes
          the profile <code>--puck-profile</code> loads. Tune at the exposure you play at:
          <code>sat_min</code>/<code>val_min</code> are fixed thresholds, so a profile fitted
          at another exposure mis-segments.</p>
        <div class="env" style="margin-bottom:10px">
          <span>Resolution</span><input id="resolution" spellcheck="false">
        </div>
        <pre class="cmd tune" id="tuneCmd"></pre>
        <p class="prose keys">In its window: <b>L-click</b> the puck &middot;
          <b>R-click</b> hand/table to reject &middot; <b>a</b> auto-fit &middot;
          <b>s</b> save &middot; <b>r</b> reset &middot; <b>q</b> quit</p>
        <button class="btn wide" id="calibrate">Calibrate detection</button>
      </div>
    </div>

    <div class="card">
      <h2>Environment</h2>
      <div class="pad env">
        <span>Directory</span><input id="cwd" spellcheck="false">
        <span>Shell prelude</span><input id="prelude" spellcheck="false"
            placeholder="leave empty to inherit this shell's environment">
      </div>
    </div>
  </div>

  <div class="col right">
    <div class="card">
      <h2>Command<span class="spacer"></span>
        <button class="btn sm" id="copy">Copy</button></h2>
      <pre class="cmd" id="cmd"></pre>
      <div id="warnings"></div>
    </div>
    <div class="card">
      <h2>Output<span class="spacer"></span>
        <label class="check"><input type="checkbox" id="follow" checked> follow</label>
        <button class="btn sm" id="clearLog">Clear</button></h2>
      <pre class="log" id="log"><span class="empty">Not started.</span></pre>
      <div class="stdin"><span>&rsaquo;</span>
        <input id="stdin" placeholder="send a line to the program (Enter)" spellcheck="false"
               disabled></div>
    </div>
  </div>
</main>

<div class="toast" id="toast"></div>

<script>
const $ = (s, r = document) => r.querySelector(s);
const el = (t, cls, txt) => { const n = document.createElement(t);
  if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };

let SPEC = [], BY = {}, SECTIONS = [], VALUES = {}, PRESETS = {}, BUILTIN = [];
let ROWS = {}, GROUP_EL = {}, PINNED = [], STATE = {status: 'idle'};
let LINES = [], BASE = 0, started = null;

const toast = (msg, bad) => { const t = $('#toast'); t.textContent = msg;
  t.className = 'toast show' + (bad ? ' bad' : '');
  clearTimeout(t._h); t._h = setTimeout(() => t.className = 'toast', 2600); };

const isSet = n => { const v = VALUES[n];
  return v === true || (typeof v === 'string' && v.trim() !== ''); };

/* ---------- build the flag rows once; mutate them afterwards ---------- */
function buildFlags() {
  const host = $('#groups'); host.textContent = '';
  for (const {title: g, blurb} of SECTIONS) {
    const flags = SPEC.filter(f => f.group === g);
    if (!flags.length) continue;
    const d = el('details', 'group'); d.dataset.group = g; d.open = g === 'Run';
    const s = el('summary'); s.append(el('span', null, g));
    const c = el('span', 'count'); c.dataset.group = g; s.append(c);
    d.append(s);
    if (blurb) d.append(el('p', 'blurb', blurb));
    flags.forEach((f, i) => {
      const row = buildRow(f);
      row.dataset.group = g; row.dataset.idx = i;     // idx restores order after unpinning
      ROWS[f.name] = row; d.append(row);
    });
    GROUP_EL[g] = d; host.append(d);
  }
}

/* A group whose flags are all at their defaults is noise; one the run actually sets is not.
   Only ever opens -- a group closed by hand stays closed until its values change. */
function autoOpenGroups() {
  for (const g in GROUP_EL)
    if (SPEC.some(f => f.group === g && isSet(f.name))) GROUP_EL[g].open = true;
}

function buildRow(f) {
  const row = el('div', 'row'); row.dataset.name = f.name;
  row.dataset.hay = (f.name + ' ' + f.help).toLowerCase();

  const lab = el('label');
  lab.append(el('span', null, '--' + f.name));
  if (f.required) lab.append(el('span', 'req', ' *'));
  lab.title = (f.help || '(no help text)') +
    (f.default !== '' ? '\n\ndefault: ' + f.default : '\n\ndefault: unset');
  row.append(lab);

  let ctl;
  if (f.kind === 'flag') {
    ctl = el('button', 'sw'); ctl.type = 'button';
    ctl.onclick = () => { setVal(f.name, VALUES[f.name] === true ? '' : true); };
  } else if (f.kind === 'tri') {
    ctl = el('div', 'seg');
    for (const [v, txt] of [['', 'auto'], ['on', 'on'], ['off', 'off']]) {
      const b = el('button', null, txt); b.type = 'button'; b.dataset.v = v;
      b.onclick = () => setVal(f.name, v); ctl.append(b);
    }
  } else if (f.kind === 'choice') {
    ctl = el('select');
    if (!f.required) ctl.append(new Option('(default: ' + (f.default || 'unset') + ')', ''));
    for (const c of f.choices) ctl.append(new Option(c, c));
    ctl.onchange = () => setVal(f.name, ctl.value);
  } else {
    ctl = el('input');
    if (f.kind === 'number') { ctl.type = 'number'; ctl.step = 'any'; }
    ctl.placeholder = f.default === '' ? 'unset' : f.default;
    ctl.spellcheck = false;
    ctl.oninput = () => setVal(f.name, ctl.value, true);
  }
  ctl.dataset.ctl = f.name;
  row.append(ctl);

  const x = el('button', 'clear', '×'); x.type = 'button';
  x.title = 'reset to default'; x.onclick = () => setVal(f.name, '');
  row.append(x);

  const pin = el('button', 'pin', '★'); pin.type = 'button';
  pin.title = 'keep this one at the top';
  pin.onclick = () => togglePin(f.name);
  row.append(pin);
  return row;
}

/* ---------- pinning ---------- */
function togglePin(name) {
  PINNED = PINNED.includes(name) ? PINNED.filter(n => n !== name) : [...PINNED, name];
  layoutPins(); applyFilter(); schedulePreview();
}
function layoutPins() {
  const host = $('#pinned');
  for (const name of PINNED) { const r = ROWS[name]; if (r) host.append(r); }
  for (const name in ROWS) {                       // return unpinned rows to their group
    const r = ROWS[name];
    if (PINNED.includes(name) || r.parentElement === GROUP_EL[r.dataset.group]) continue;
    const g = GROUP_EL[r.dataset.group], idx = +r.dataset.idx;
    const after = [...g.children].find(c => c.dataset.idx !== undefined && +c.dataset.idx > idx);
    g.insertBefore(r, after || null);
  }
  for (const name in ROWS) ROWS[name].classList.toggle('pinned', PINNED.includes(name));
}

function setVal(name, v, fromInput) {
  VALUES[name] = v;
  syncRow(name, fromInput);
  if (name === 'goalie')                       // it implies a whole group of other flags
    SPEC.filter(f => f.goalie_default).forEach(f => syncRow(f.name));
  updateCounts();
  schedulePreview();
}

function syncRow(name, fromInput) {
  const f = BY[name], row = $(`.row[data-name="${name}"]`);
  if (!row) return;
  const v = VALUES[name] ?? '';
  row.classList.toggle('set', isSet(name));
  const ctl = $(`[data-ctl="${name}"]`, row);

  // --goalie fills in a whole group of flags that are left unset. Show what it will do,
  // so an untouched row does not read as "off".
  const implied = (VALUES['goalie'] === true && f.goalie_default && !isSet(name))
                  ? f.goalie_default : null;
  row.classList.toggle('bygoalie', !!implied);
  if (ctl.placeholder !== undefined)
    ctl.placeholder = implied ? implied : (f.default === '' ? 'unset' : f.default);

  if (f.kind === 'flag') ctl.classList.toggle('on', v === true);
  else if (f.kind === 'tri')
    [...ctl.children].forEach(b => {
      b.classList.toggle('on', b.dataset.v === (v || ''));
      b.classList.toggle('neg', b.dataset.v === 'off');
    });
  else if (!fromInput) ctl.value = v === true ? '' : v;
}

function updateCounts() {
  for (const c of document.querySelectorAll('.count')) {
    const n = SPEC.filter(f => f.group === c.dataset.group && isSet(f.name)).length;
    const total = SPEC.filter(f => f.group === c.dataset.group).length;
    c.textContent = n ? `${n} / ${total}` : total;
    c.classList.toggle('on', n > 0);
  }
}

function syncAll() { SPEC.forEach(f => syncRow(f.name)); updateCounts(); applyFilter(); }

/* ---------- filter ---------- */
function applyFilter() {
  const q = $('#search').value.trim().toLowerCase(), only = $('#onlySet').checked;
  for (const name in ROWS) {
    const row = ROWS[name];
    row.hidden = !((!q || row.dataset.hay.includes(q)) && (!only || isSet(name)));
  }
  for (const g in GROUP_EL) {
    const d = GROUP_EL[g];
    const shown = [...d.querySelectorAll('.row')].some(r => !r.hidden);
    d.hidden = !shown;
    if ((q || only) && shown) d.open = true;
  }
  $('#pinnedCard').hidden = !PINNED.some(n => ROWS[n] && !ROWS[n].hidden);
}

/* ---------- command preview ---------- */
let previewTimer = null;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPreview, 180);
}
async function refreshPreview() {
  const r = await fetch('/api/preview', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({values: VALUES, cwd: $('#cwd').value,
                          prelude: $('#prelude').value, preset: $('#presetSel').value,
                          resolution: $('#resolution').value, pinned: PINNED})});
  const d = await r.json();
  $('#cmd').textContent = d.cmd;
  $('#tuneCmd').textContent = d.tune_cmd;
  const w = $('#warnings'); w.textContent = '';
  for (const msg of d.warnings) w.append(el('div', 'warn', msg));
}

/* ---------- presets ---------- */
function fillPresets(selected) {
  const sel = $('#presetSel'); sel.textContent = '';
  for (const name of Object.keys(PRESETS))
    sel.append(new Option(name + (BUILTIN.includes(name) ? '  (built-in)' : ''), name));
  sel.value = selected in PRESETS ? selected : Object.keys(PRESETS)[0] ?? '';
}
function applyPreset(name) {
  if (!PRESETS[name]) return;
  VALUES = {...PRESETS[name]};
  syncAll(); autoOpenGroups(); refreshPreview();
}

/* ---------- log ----------
   One DOM node per line, updated in place. A full rebuild per incoming line janks badly
   once a long run has filled the buffer. */
const noteClass = t =>
  (t.startsWith('$ ') || t.startsWith('— ') || t.startsWith('> ')) ? 'note' : '';
const follow = () => {
  if ($('#follow').checked) { const l = $('#log'); l.scrollTop = l.scrollHeight; } };

function renderLog() {
  const log = $('#log');
  if (!LINES.length) { log.innerHTML = '<span class="empty">Not started.</span>'; return; }
  const frag = document.createDocumentFragment();
  for (const line of LINES) frag.append(el('div', noteClass(line), line || ' '));
  log.textContent = ''; log.append(frag);
  follow();
}
function pushLine(i, t) {
  const pos = i - BASE, log = $('#log');
  if (pos === LINES.length) {
    if (!LINES.length) log.textContent = '';          // drop the "Not started." placeholder
    LINES.push(t); log.append(el('div', noteClass(t), t || ' '));
  } else if (pos >= 0 && pos < LINES.length) {
    LINES[pos] = t;
    const node = log.children[pos];
    if (!node) return renderLog();
    node.textContent = t || ' '; node.className = noteClass(t);
  } else return;
  if (LINES.length > 9000) { LINES.splice(0, 2000); BASE += 2000; renderLog(); }
  else follow();
}

/* ---------- status ---------- */
function setState(s) {
  STATE = s;
  started = s.started_at;
  const pill = $('#status');
  pill.className = 'pill ' + s.status +
    (s.status === 'exited' && s.exit_code === 0 ? ' clean' : '');
  const run = s.status === 'running' || s.status === 'stopping';
  $('#start').disabled = run;
  $('#stop').disabled = !run;
  $('#kill').hidden = !s.can_force;
  $('#stdin').disabled = !run;
  $('#calibrate').disabled = run;      // one process at a time, and both want the camera
  tick();
}
function tick() {
  let txt = STATE.status;
  if (STATE.status === 'exited' && STATE.exit_code != null)
    txt = 'exited ' + STATE.exit_code;
  if (started) {
    const d = Math.max(0, Math.floor(Date.now() / 1000 - started));
    txt += '  ' + String(Math.floor(d / 60)).padStart(2, '0') +
           ':' + String(d % 60).padStart(2, '0');
  }
  $('#statusText').textContent = txt;
}
setInterval(() => {
  tick();
  // Nothing pushes an event when the graceful-stop grace period lapses, so while a stop is
  // in flight ask, once a second, whether force-kill should be offered yet.
  if (STATE.status === 'stopping' && !STATE.can_force)
    fetch('/api/state').then(r => r.json()).then(setState).catch(() => {});
}, 1000);

/* ---------- wiring ---------- */
async function launch(tool) {
  const r = await fetch('/api/start', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({values: VALUES, cwd: $('#cwd').value, tool,
                          prelude: $('#prelude').value,
                          resolution: $('#resolution').value})});
  const d = await r.json();
  if (d.error) toast(d.error, true);
}
$('#start').onclick = () => launch(null);
$('#calibrate').onclick = () => launch('tune');
$('#stop').onclick = () => fetch('/api/stop', {method: 'POST',
  headers: {'Content-Type': 'application/json'}, body: '{}'});
$('#kill').onclick = () => {
  if (confirm('SIGKILL skips the shutdown that saves --log telemetry and hands the arm ' +
              'to gravity compensation.\n\nForce kill anyway?'))
    fetch('/api/stop', {method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({force: true})});
};
$('#copy').onclick = async () => {
  try { await navigator.clipboard.writeText($('#cmd').textContent); toast('Command copied'); }
  catch { toast('Copy failed — select it by hand', true); }
};
$('#clearLog').onclick = () => { LINES = []; BASE = 0; renderLog(); };
$('#stdin').onkeydown = e => {
  if (e.key !== 'Enter') return;
  const text = e.target.value; e.target.value = '';
  fetch('/api/stdin', {method: 'POST', headers: {'Content-Type': 'application/json'},
                       body: JSON.stringify({text})});
};
$('#search').oninput = applyFilter;
$('#onlySet').onchange = applyFilter;
$('#collapse').onclick = () => {
  const any = [...document.querySelectorAll('details.group')].some(d => d.open);
  document.querySelectorAll('details.group').forEach(d => d.open = !any);
  $('#collapse').textContent = any ? 'Expand all' : 'Collapse all';
};
$('#reset').onclick = () => {
  if (!confirm('Clear every flag?')) return;
  VALUES = {}; syncAll(); refreshPreview();
};
$('#presetSel').onchange = e => applyPreset(e.target.value);
$('#presetSave').onclick = async () => {
  const name = prompt('Save the current flags as:', $('#presetSel').value);
  if (!name) return;
  const r = await fetch('/api/presets', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, values: VALUES})});
  const d = await r.json();
  PRESETS = d.presets; BUILTIN = d.builtin; fillPresets(name); toast('Saved "' + name + '"');
};
$('#presetDel').onclick = async () => {
  const name = $('#presetSel').value;
  if (BUILTIN.includes(name)) return toast('Built-in presets cannot be deleted', true);
  if (!confirm('Delete preset "' + name + '"?')) return;
  const r = await fetch('/api/presets', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, delete: true})});
  const d = await r.json();
  PRESETS = d.presets; BUILTIN = d.builtin; fillPresets(''); toast('Deleted');
};
$('#cwd').oninput = schedulePreview;
$('#prelude').oninput = schedulePreview;
$('#resolution').oninput = schedulePreview;

/* ---------- boot ---------- */
(async () => {
  const d = await (await fetch('/api/spec')).json();
  SPEC = d.flags; BY = Object.fromEntries(SPEC.map(f => [f.name, f]));
  SECTIONS = d.groups; PINNED = d.pinned || [];
  PRESETS = d.presets; BUILTIN = d.builtin; VALUES = {...d.values};
  $('#mod').textContent = d.module;
  $('#cwd').value = d.cwd; $('#prelude').value = d.prelude;
  $('#resolution').value = d.resolution;
  buildFlags(); layoutPins(); fillPresets(d.preset);
  syncAll(); autoOpenGroups(); refreshPreview();

  const es = new EventSource('/api/events');
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === 'snapshot') {
      LINES = m.lines; BASE = m.base; renderLog(); setState(m.state);
    } else if (m.type === 'line') pushLine(m.i, m.t);
    else if (m.type === 'state') setState(m.state);
  };
})();
</script>
</body></html>
"""


# --------------------------------------------------------------------------------------

def _default_prelude() -> str:
    parts = []
    ros = Path("/opt/ros/humble/setup.bash")
    if ros.exists():
        parts.append(f"source {ros}")
    conda_sh = None
    exe = os.environ.get("CONDA_EXE")
    if exe:
        cand = Path(exe).resolve().parents[1] / "etc" / "profile.d" / "conda.sh"
        if cand.exists():
            conda_sh = cand
    if conda_sh is None:
        cand = Path.home() / "miniconda3" / "etc" / "profile.d" / "conda.sh"
        if cand.exists():
            conda_sh = cand
    if conda_sh is not None:
        parts.append(f"source {conda_sh} && conda activate metal_hw")
    return " && ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to reach the GUI from outside the container")
    ap.add_argument("--cwd", default=str(REPO_ROOT),
                    help="working directory for the run (where the .npz files live)")
    ap.add_argument("--prelude", default=None,
                    help="shell line sourced before the run; '' inherits this environment")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    Handler.spec = parse_cli_spec(CLI_PATH)
    Handler.cwd = args.cwd
    Handler.prelude = _default_prelude() if args.prelude is None else args.prelude

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}"
    print(f"match_gui: {len(Handler.spec)} flags from {CLI_PATH.name}")
    print(f"match_gui: runs in {args.cwd}")
    print(f"match_gui: open {url}   (Ctrl-C here stops the GUI, and any live run with it)")

    if not args.no_browser and os.environ.get("DISPLAY"):
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nmatch_gui: shutting down")
    finally:
        RUNNER.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
