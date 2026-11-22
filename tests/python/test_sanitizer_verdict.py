"""Tests for scripts/sanitizer_verdict.py on log shapes seen on this machine.

Regression test for a classifier bug: racecheck prints a
"RACECHECK SUMMARY" line instead of "ERROR SUMMARY", and the old classifier
reported a clean racecheck run as FAILED.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from sanitizer_verdict import classify  # noqa: E402

DONE = "[  OK  ] something\n0 of 10 tests failed, 1 skipped, 0 cases skipped by --max-n\n"
BLOCKED_PREFIX = ("========= Error: Failed to initialize WDDM debugger interface\n"
                  "========= Error: Device not supported\n")


def test_memcheck_clean():
    assert classify("========= COMPUTE-SANITIZER\n" + DONE + "========= ERROR SUMMARY: 0 errors\n", 0)[0] == "CLEAN"


def test_racecheck_clean_uses_its_own_summary_line():
    text = DONE + "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
    assert classify(text, 0)[0] == "CLEAN"


def test_racecheck_hazard_is_errors():
    text = ("========= Error: Race reported between Write access at k+0x10\n" + DONE +
            "========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)\n")
    assert classify(text, 0)[0] == "ERRORS"


def test_initcheck_report_is_errors():
    text = ("========= Uninitialized __global__ memory read of size 4 bytes\n" + DONE +
            "========= ERROR SUMMARY: 64 errors\n")
    verdict, detail = classify(text, 99)
    assert verdict == "ERRORS" and "Uninitialized" in detail


def test_disabled_debugger_interface_is_blocked():
    text = BLOCKED_PREFIX + DONE + "========= ERROR SUMMARY: 2 errors\n"
    assert classify(text, 99)[0] == "BLOCKED"


def test_abnormal_exit_is_failed_even_if_blocked():
    text = BLOCKED_PREFIX + DONE + "========= Error: process didn't terminate successfully\n"
    assert classify(text, 99)[0] == "FAILED"


def test_test_failure_is_failed():
    assert classify("1 of 10 tests failed\n========= ERROR SUMMARY: 0 errors\n", 1)[0] == "FAILED"


def test_missing_summary_is_failed():
    assert classify(DONE, 0)[0] == "FAILED"
