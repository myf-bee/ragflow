#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Shared setup for RAGFlow unit tests.

Several parsers and the chunking pipeline tokenize text with NLTK, which needs
the ``punkt_tab`` and ``wordnet`` data sets. Production provisions these via
``download_deps.py`` (into ``nltk_data``, exported as ``NLTK_DATA`` by
``docker/launch_backend_service.sh``) and ``api.validation`` at startup, but the
unit-test runner has neither. Without the data, tokenizer-backed tests such as
``test_epub_parser`` and ``test_dataflow_service`` fail with
``LookupError: Resource 'punkt_tab' not found``. Make sure the data is reachable
before any test imports a tokenizer: reuse a provisioned ``nltk_data`` directory
when present, and download only what is still missing.
"""

import os
import time

import nltk

# Reuse data already fetched by download_deps.py (the directory the app exports
# as NLTK_DATA) so provisioned environments do not download it again.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_LOCAL_NLTK_DATA = os.path.join(_REPO_ROOT, "ragflow_deps", "nltk_data")
if os.path.isdir(_LOCAL_NLTK_DATA) and _LOCAL_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, _LOCAL_NLTK_DATA)

# (download name, resource path used by nltk.data.find)
_REQUIRED_NLTK_DATA = (
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("wordnet", "corpora/wordnet"),
)
for _name, _find_path in _REQUIRED_NLTK_DATA:
    try:
        nltk.data.find(_find_path)
    except LookupError:
        nltk.download(_name, quiet=True)


# Diagnostic collection logging. pytest is silent during its collection phase
# (importing test modules and the heavy app code they pull in), so CI hangs
# here show nothing until "collected N items". When RAGFLOW_TEST_COLLECT_LOG=1
# we print a start marker for every test module as pytest begins importing it:
# the last module printed before a long silence is the one pytest is stuck on.
# At the end we also print the slowest modules ranked by collection time.
_COLLECT_LOG = os.environ.get("RAGFLOW_TEST_COLLECT_LOG") == "1"
_COLLECT_START = {}  # nodeid -> monotonic start time
_COLLECT_MODULE_TIMES = []  # (nodeid, duration) for the final summary


def pytest_sessionstart(session):
    if _COLLECT_LOG:
        print("[COLLECT] session collection started", flush=True)


def pytest_collectstart(collector):
    if not _COLLECT_LOG:
        return
    path = getattr(collector, "path", None)
    if path is None:
        return
    try:
        is_dir = path.is_dir()
    except AttributeError:
        is_dir = bool(getattr(path, "isdir", lambda: False)())
    if is_dir:
        # Directory/Package boundary, kept for orientation in the CI log.
        print(f"[COLLECT] {path}", flush=True)
    else:
        # Per-module start marker: the last one before a long silence is the
        # module pytest is stuck importing.
        nodeid = getattr(collector, "nodeid", None)
        if nodeid:
            _COLLECT_START[nodeid] = time.monotonic()
        print(f"[COLLECT-START] {path}", flush=True)


def pytest_collectreport(report):
    if not _COLLECT_LOG:
        return
    nodeid = report.nodeid
    # Only consider module (file) collectors, not directories or the session.
    if not nodeid or nodeid.endswith("/") or not nodeid.endswith(".py"):
        return
    # CollectReport.duration is always 0 for the collection phase in pytest,
    # so measure wall-clock time ourselves from the start marker recorded in
    # pytest_collectstart. .pop() also releases the entry after use.
    start = _COLLECT_START.pop(nodeid, None)
    if start is not None:
        dur = time.monotonic() - start
    else:
        dur = getattr(report, "duration", None) or 0.0
    _COLLECT_MODULE_TIMES.append((nodeid, dur))
    if dur >= 2.0:
        print(f"[COLLECT-SLOW] {nodeid} {dur:.2f}s", flush=True)


def pytest_collection_modifyitems(session, config, items):
    if not _COLLECT_LOG:
        return
    print(f"[COLLECT] session collection finished: {len(items)} items", flush=True)
    if _COLLECT_MODULE_TIMES:
        top = sorted(_COLLECT_MODULE_TIMES, key=lambda x: x[1], reverse=True)[:15]
        print("[COLLECT] slowest modules:", flush=True)
        for nodeid, dur in top:
            print(f"[COLLECT]   {dur:8.2f}s  {nodeid}", flush=True)
