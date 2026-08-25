# legacy

The acquired system and the documents written about it. **Nothing here runs, and nothing in
`app/` or `engine/` imports it.** It is kept because the rebuild's design decisions only make
sense against what they were reacting to, and a claim that eighteen defects were fixed is
worth more when the code containing them is still readable.

| File | What it is |
| --- | --- |
| `acquired-system-v1.py` | The acquired system: 1,146 lines in one file, holding the interface, the optimiser, the simulation, the price assumptions, and the authentication together. Defects F1–F18 in [`../engine/docs/ADR-001-objective-reformulation.md`](../engine/docs/ADR-001-objective-reformulation.md) are all citations into this file. |
| `forensic-audit-v1.txt` | The audit of that system, separating what was implemented from what was claimed but never demonstrated. |
| `rebuild-blueprint.pdf` | The target architecture the rebuild was aimed at. |
| `system-map-sketch.png` | Working sketch of the system map. |

The parity constants in the test suite exist to prove the rebuild reproduces this system's
*correct* behaviour while fixing the rest, so this file is a specification as much as a
historical artifact. `engine/src/scrcae/legacy/reference.py` is where that parity is encoded —
that module is real, tested code, and is not the same thing as this directory.

Original filenames were `Best poop.py`, `Forensic audit.txt`, `Supply Chain Resilience &
Capital Allocation Engine — Rebuild Blueprint.pdf`, and a dated screenshot. Renamed on the
way in, because filenames in a repository are read by people deciding whether to take it
seriously.
