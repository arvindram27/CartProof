"""Narrow root .env loader for the fixed CartProof Meta smoke (offline).

Loads exactly ``PROJECT_ROOT / ".env"`` — never the current working directory,
never an upward search, and never a general dotenv dialect. Called only after
every explicit budget/model/origin/reasoning gate has passed and the key is
absent from the already-chosen environment mapping. Nothing is read while a
callback is constructed or a module is imported, and ``os.environ`` is never
mutated.

Residual limit, stated plainly: stdlib symlink/junction checks plus ``fstat``
cannot make concurrent hostile path replacement race-proof on Windows. This is
a small local safeguard, not race-free hostile-filesystem protection.
"""
from __future__ import annotations

import codecs
import os
import stat
import unicodedata
from pathlib import Path
from typing import Callable, Optional

_MAX_BYTES = 16384
_KEY = "MODEL_API_KEY"


class DotEnvError(ValueError):
    """Sanitized loader failure. ``str(exc)`` is a fixed category only."""


def _is_link_like(p: Path) -> bool:
    try:
        if p.is_symlink():
            return True
        is_junction = getattr(p, "is_junction", None)
        return bool(is_junction and is_junction())
    except OSError:
        return True


def _check_lexical_components(anchor_lexical: Path, target_lexical: Path) -> None:
    """Reject symlink/junction components BEFORE resolve() hides them.

    Walks every component from the filesystem root through the anchor and
    the target, preserving the lexical path the caller actually supplied.
    """
    # Anchor chain from the filesystem root down (drive/root included via
    # cumulative joins over anchor_lexical.parts).
    parts = anchor_lexical.parts
    current = Path(parts[0]) if parts else anchor_lexical
    if _is_link_like(current):
        raise DotEnvError("dot_env_unsafe_path")
    for part in parts[1:]:
        current = current / part
        if _is_link_like(current):
            raise DotEnvError("dot_env_unsafe_path")
    if _is_link_like(target_lexical):
        raise DotEnvError("dot_env_unsafe_path")
    try:
        target_lexical.relative_to(anchor_lexical)
    except ValueError:
        raise DotEnvError("dot_env_unsafe_path") from None


def load_root_dotenv(anchor: Path, key: str = _KEY) -> str:
    """Return the single ``MODEL_API_KEY`` value from ``anchor / ".env"``.

    Raises DotEnvError with a fixed sanitized category on any failure. Never
    echoes a line, value, or key fragment, and chains no raw exception.
    """
    anchor_lexical = Path(os.path.abspath(str(anchor)))
    target_lexical = anchor_lexical / ".env"
    _check_lexical_components(anchor_lexical, target_lexical)

    resolved_root = anchor_lexical.resolve()
    resolved_target = target_lexical.resolve()
    if resolved_target.parent != resolved_root \
            or not resolved_target.is_relative_to(resolved_root):
        raise DotEnvError("dot_env_unsafe_path")

    # Pre-open metadata (no-follow) and opened-handle identity must agree.
    try:
        pre = os.lstat(target_lexical)
    except FileNotFoundError:
        raise DotEnvError("dot_env_missing") from None
    except OSError:
        raise DotEnvError("dot_env_read_error") from None
    if stat.S_ISLNK(pre.st_mode):
        raise DotEnvError("dot_env_unsafe_path")
    try:
        f = open(target_lexical, "rb")
    except FileNotFoundError:
        raise DotEnvError("dot_env_missing") from None
    except OSError:
        raise DotEnvError("dot_env_read_error") from None
    close_failed = False
    try:
        try:
            st = os.fstat(f.fileno())
        except OSError:
            raise DotEnvError("dot_env_read_error") from None
        if not stat.S_ISREG(st.st_mode) \
                or (st.st_dev, st.st_ino) != (pre.st_dev, pre.st_ino):
            raise DotEnvError("dot_env_unsafe_path")
        raw = f.read(_MAX_BYTES + 1)
    except DotEnvError:
        raise
    except OSError:
        raise DotEnvError("dot_env_read_error") from None
    finally:
        try:
            f.close()
        except OSError:
            close_failed = True
    if close_failed:
        raise DotEnvError("dot_env_read_error")

    if len(raw) > _MAX_BYTES:
        raise DotEnvError("dot_env_too_large")
    if raw.startswith(codecs.BOM_UTF8):
        raw = raw[len(codecs.BOM_UTF8):]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise DotEnvError("dot_env_invalid_encoding") from None

    return _parse(text, key)


_FORBIDDEN_VALUE_CHARS = set(" \t\"'`#$\\")


def _parse(text: str, key: str) -> str:
    assignments: list[tuple[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in line:
            raise DotEnvError("dot_env_invalid_syntax")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise DotEnvError("dot_env_invalid_syntax")
        if any(c.isspace() for c in value) \
                or any(unicodedata.category(c) == "Cc" for c in value) \
                or any(c in _FORBIDDEN_VALUE_CHARS for c in value):
            raise DotEnvError("dot_env_invalid_syntax")
        assignments.append((name, value))

    if not assignments:
        raise DotEnvError("dot_env_missing")
    if any(name != key for name, _v in assignments):
        raise DotEnvError("dot_env_invalid_syntax")
    if len(assignments) != 1:
        raise DotEnvError("dot_env_invalid_syntax")
    value = assignments[0][1]
    if not value:
        raise DotEnvError("dot_env_empty")
    return value


def make_dotenv_fallback(anchor: Path, key: str = _KEY) -> Callable[[], str]:
    """Zero-I/O lazy callback: constructing it reads nothing.

    Invoking it loads the root .env once and returns the credential privately
    to the caller. The anchor is captured lexically, never serialized.
    """
    captured = Path(anchor)

    def _load() -> str:
        return load_root_dotenv(captured, key)

    return _load
