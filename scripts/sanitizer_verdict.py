"""Verdict for one Compute Sanitizer log (used by scripts/run-sanitizer.ps1).

Checked in this order:
  FAILED   the test program did not complete with "0 of N tests failed"
           (crash, abnormal exit, test failure) or gpu_guard stopped it
  ERRORS   the log reports a sanitizer error or hazard other than the two
           start-up messages a disabled WDDM debugger interface produces
  BLOCKED  the program completed and the only errors are those start-up
           messages: kernels were NOT checked
  CLEAN    the tool ran with device-side checking, exited 0, and its
           summary line reports nothing ("ERROR SUMMARY: 0 errors" for
           memcheck / initcheck / synccheck, "RACECHECK SUMMARY: 0 hazards
           displayed (0 errors, 0 warnings)" for racecheck)
  FAILED   anything else (no summary line, nonzero exit)

Usage: python scripts/sanitizer_verdict.py LOG EXIT_CODE  ->  prints "VERDICT<TAB>detail"
"""

from __future__ import annotations

import re
import sys

KNOWN_STARTUP = re.compile(r"Failed to initialize WDDM debugger interface|Error: Device not supported")
REPORT = re.compile(r"^=+\s+(Error|CUDA API Error|Program hit|Invalid|Uninitialized|Barrier|Warning|Leaked)")


def classify(text: str, exit_code: int) -> tuple[str, str]:
    completed = (re.search(r"(?m)^0 of \d+ tests failed", text) is not None
                 and "Target application returned an error" not in text
                 and "process didn't terminate successfully" not in text
                 and "SAFETY STOP" not in text)
    if not completed:
        return "FAILED", "the test program did not complete cleanly"
    lines = text.splitlines()
    known = [ln for ln in lines if re.match(r"^=+\s+Error: ", ln) and KNOWN_STARTUP.search(ln)]
    # Case-sensitive on purpose: the summary lines themselves are not reports.
    other = [ln.strip() for ln in lines if REPORT.match(ln) and not KNOWN_STARTUP.search(ln) and "SUMMARY" not in ln]
    err = re.search(r"ERROR SUMMARY: (\d+) error", text)
    race = re.search(r"RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)", text)
    reported = None
    if err:
        reported = int(err.group(1))
    elif race:
        reported = int(race.group(2)) + int(race.group(3))
    if reported is not None and reported != len(known):
        other.append(f"summary reports {reported} errors/hazards, only {len(known)} are the known start-up messages")
    if other:
        return "ERRORS", other[0]
    if "Failed to initialize WDDM debugger interface" in text:
        return "BLOCKED", "WDDM debugger interface disabled: kernels were not checked"
    if exit_code == 0 and reported == 0:
        return "CLEAN", "device-side checking ran; 0 errors reported"
    return "FAILED", "no clean summary line or nonzero exit"


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8-sig", errors="replace") as fh:
        verdict, detail = classify(fh.read(), int(sys.argv[2]))
    print(f"{verdict}\t{detail}")
