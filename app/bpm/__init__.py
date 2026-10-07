"""Automatic BPM detection.

The implementation lives in :mod:`app.bpm.detect`.  Import it as a module::

    from .bpm import detect as bpm_module        # app.pipeline, app.api.routes
    import app.bpm.detect as bpm_module          # equivalent

Note: this package deliberately does **not** re-export the functions from
``detect`` (e.g. ``from .detect import detect_bpm``).  Doing so would bind the
name ``detect`` in this namespace to the *function*, shadowing the submodule and
breaking ``import app.bpm.detect``.
"""

from __future__ import annotations

__all__ = ["detect"]
