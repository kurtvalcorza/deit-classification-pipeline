"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import importlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/deit_classification_colab.ipynb"
BYOD_CELL = "deit_classification-33"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


SOURCE = _cell(BYOD_CELL)
PIPELINE_CLASS = re.search(r"byod_pipe = (\w+)\.from_pretrained", SOURCE).group(1)
_PACKAGE = next(p.name for p in (ROOT / "src").iterdir() if (p / "data.py").is_file())
DATA, PIPELINE = importlib.import_module(f"{_PACKAGE}.data"), importlib.import_module(f"{_PACKAGE}.pipeline")
EARLIER = {"byod_image": "old", "byod_result": "old", "byod_manifest": "old", "byod_after": "old", "byod_export": "old"}


def _png(value: int = 120, size=(32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (value, (value * 7) % 256, (value * 13) % 256)).save(buffer, format="PNG")
    return buffer.getvalue()


def _seed_outputs(tmp_path: Path) -> None:
    byod = tmp_path / "outputs" / "byod"
    byod.mkdir(parents=True, exist_ok=True)
    for name in ("byod_x_adapter.safetensors", "byod_x_classification_result.json", "byod_x_classification_predictions.csv"):
        (byod / name).write_text("{}", encoding="utf-8")
    (tmp_path / "outputs" / "sample_result.json").write_text("{}", encoding="utf-8")


class _Pipeline:
    @classmethod
    def from_pretrained(cls, **kwargs):
        raise RuntimeError("internal: weights not staged")


def _namespace(pipe=None) -> dict:
    ns: dict = {k: v for module in (PIPELINE, DATA) for k, v in vars(module).items() if not k.startswith("__")}
    ns.update({
        "json": json, "Path": Path, "Image": Image, "OUTPUTS": Path("outputs"), "TOP_K": 5, "EPOCHS": 1, "SEED": 0, "BATCH_SIZE": 2,
        "LEARNING_RATE": 1e-4, "FREEZE_BACKBONE": True, "MAX_BATCH": 4, "WEIGHTS_DIR": Path("weights"), "SAMPLE_CLASSES": ["frog", "truck"],
        PIPELINE_CLASS: _Pipeline, "pipe": pipe or types.SimpleNamespace(predict=lambda image, top_k: {"predictions": [{"top_k": [{"label": "tench, x", "score": 0.5}]}]}),
        "evaluation_report": lambda *a, **k: {"verdict": "not-measurable"},
    })
    for name in ("blank_image", "noise_image"):
        ns.setdefault(name, lambda *a, **k: Image.new("RGB", (32, 32)))
    ns.update(EARLIER)
    return ns


def _run(monkeypatch, tmp_path, *, image=None, dataset=None, use_image=False, use_dataset=False, colab=None, pipe=None) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    _seed_outputs(tmp_path)
    source = SOURCE.replace("USE_BYOD_IMAGE = False", f"USE_BYOD_IMAGE = {use_image}", 1).replace("USE_BYOD_DATASET = False", f"USE_BYOD_DATASET = {use_dataset}", 1)
    if image is not None:
        source = source.replace('BYOD_IMAGE_PATH = ""', f"BYOD_IMAGE_PATH = {image!r}", 1)
    if dataset is not None:
        source = source.replace('BYOD_DATASET_PATH = ""', f"BYOD_DATASET_PATH = {dataset!r}", 1)
    ns = _namespace(pipe)
    exec(compile(source, "<section 13>", "exec"), ns)
    return ns


def _colab(uploaded: dict):
    module = types.ModuleType("google.colab")
    module.files = types.SimpleNamespace(upload=lambda: uploaded)
    return module


def _assert_refused(caught, file_name, rule, tmp_path):
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs" / "byod").glob("byod_*")) == []
    assert (tmp_path / "outputs" / "sample_result.json").exists()  # the sample's results are not this input's


# --- image branch -------------------------------------------------------------------------------------------------


def _image_cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime-host\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    return {
        "text file (the scripted /etc/hostname run)": ("hostname", "not an image PIL can open"),
        "folder": ("folder", "does not exist or is not a file"),
        "missing path": ("nope.png", "does not exist or is not a file"),
    }


@pytest.mark.parametrize("case", ["text file (the scripted /etc/hostname run)", "folder", "missing path"])
def test_image_problems_are_one_refused_line_and_remove_earlier_byod_results(monkeypatch, tmp_path, case):
    file_name, rule = _image_cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, image=str(tmp_path / file_name), use_image=True)
    _assert_refused(caught, file_name, rule, tmp_path)


def test_image_refusal_clears_the_session(monkeypatch, tmp_path):
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    source = SOURCE.replace("USE_BYOD_IMAGE = False", "USE_BYOD_IMAGE = True", 1).replace('BYOD_IMAGE_PATH = ""', "BYOD_IMAGE_PATH = 'notes.txt'", 1)
    ns = _namespace()
    ns["pipe"] = "kept"
    with pytest.raises(ValueError, match="^notes.txt: not an image PIL can open"):
        exec(compile(source, "<section 13>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier BYOD run's results survived the refusal"
    assert ns["pipe"] == "kept"


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, use_image=True, colab=None)
    _assert_refused(caught, "(no file)", "BYOD_IMAGE_PATH is empty and this runtime has no Colab upload dialog", tmp_path)


def test_cancelled_image_upload_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, use_image=True, colab=_colab({}))
    _assert_refused(caught, "(no file)", "nothing was uploaded", tmp_path)


def test_good_image_still_runs_prediction(monkeypatch, tmp_path):
    (tmp_path / "ok.png").write_bytes(_png(size=(64, 48)))
    ns = _run(monkeypatch, tmp_path, image=str(tmp_path / "ok.png"), use_image=True)
    assert ns["byod_image"].size == (64, 48) and ns["byod_result"]["predictions"]


def test_image_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "ok.png").write_bytes(_png(size=(64, 48)))

    def broken(image, top_k):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run(monkeypatch, tmp_path, image=str(tmp_path / "ok.png"), use_image=True, pipe=types.SimpleNamespace(predict=broken))
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / "byod" / "byod_x_classification_result.json").exists()  # nothing is removed for an internal error


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD_IMAGE=True;BYOD_IMAGE_PATH='/etc/hostname': a file present in every fresh runtime."""
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, image="/etc/hostname", use_image=True)
    _assert_refused(caught, "hostname", "not an image PIL can open", tmp_path)


def test_default_run_is_unchanged(monkeypatch, tmp_path, capsys):
    ns = _run(monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert "BYOD image branch is off" in out and "BYOD dataset branch is off" in out
    assert ns["byod_export"] == "old"


# --- dataset branch -----------------------------------------------------------------------------------------------


def _zip(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


GOOD = {f"{c}/{i}.png": _png(v + 17 * i) for c, v in (("cats", 10), ("dogs", 140)) for i in range(6)}


def _dataset_cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "fake.zip").write_bytes(b"not a zip")
    (tmp_path / "one.zip").write_bytes(_zip({"cats/a.png": _png(1), "cats/b.png": _png(2)}))
    (tmp_path / "empty").mkdir()
    return {
        "not a .zip": ("hostname", "not a .zip"),
        "corrupt zip": ("fake.zip", "BadZipFile"),
        "one class": ("one.zip", "needs at least 2 class folders"),
        "no class folders": ("empty", "holds no image inside a class folder"),
        "missing": ("nope", "does not exist"),
    }


@pytest.mark.parametrize("case", ["not a .zip", "corrupt zip", "one class", "no class folders", "missing"])
def test_dataset_problems_are_one_refused_line_and_remove_earlier_byod_results(monkeypatch, tmp_path, case):
    file_name, rule = _dataset_cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, dataset=str(tmp_path / file_name), use_dataset=True)
    _assert_refused(caught, file_name, rule, tmp_path)


def test_cancelled_dataset_upload_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, use_dataset=True, colab=_colab({}))
    _assert_refused(caught, "(no file)", "nothing was uploaded", tmp_path)


def test_dataset_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, use_dataset=True, colab=None)
    _assert_refused(caught, "(no file)", "BYOD_DATASET_PATH is empty and this runtime has no Colab upload dialog", tmp_path)


def test_valid_dataset_reaches_the_pipeline_and_a_genuine_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "ok.zip").write_bytes(_zip(GOOD))
    with pytest.raises(RuntimeError, match="internal: weights not staged") as caught:
        _run(monkeypatch, tmp_path, dataset=str(tmp_path / "ok.zip"), use_dataset=True)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / "byod" / "byod_x_classification_result.json").exists()
