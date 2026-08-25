"""Turning a sentence into a reviewable proposal, and nothing else.

The acquired system had a copilot chat that read a sentence, wrote directly into
`st.session_state` (`val_total_budget`, `val_opt_weight`, `val_target_mode`,
`val_target_risk_goal`) and called `st.rerun()`. Every number on screen then moved,
with no record of what had been asked for, no way back, and no distinction between
an instruction it had understood and one it had guessed at. Its fallback for a
missing scale was `target_val if target_val > 1000 else target_val * 1_000_000`, so
"set the budget to 900" bought a nine-hundred-dollar plan and "set the budget to
1100" bought an eleven-hundred-dollar one — a discontinuity nobody could see. It also
switched target mode on for any sentence containing "risk" and a number, which is how
a user reached the structurally unattainable 20% target of F14 by typing one line.

So this module proposes. It parses the instruction into an explicit set of `Change`
records, each naming the field, the value now, the value proposed and why that
follows from what was said, and hands them back for a human to accept or reject.
Mutation is somebody else's job: `apply_proposal` returns a new dict and the caller
decides what to do with it.

Three rules hold this module in place:

* **It imports no `streamlit` and touches no session state.** Same reason the rest
  of `app` is testable without a web framework installed: parsing intent is a pure
  function of a sentence and a snapshot.
* **It solves nothing and asks the network nothing.** `interpret` is deterministic
  plain parsing — no model call, no engine call, no clock. The same sentence against
  the same snapshot always yields the same proposal, which is what makes a proposal
  reviewable at all.
* **It never guesses.** An instruction it does not recognise produces an empty
  proposal with the clause recorded in `unresolved`. This is F4 discipline — the
  audit tab that printed "Zero fractional violations detected" as a string literal —
  applied to intent: a copilot that invents a plausible change is asserting something
  it has not established, and the user pays for it in numbers they cannot trace back
  to anything they said.

Feasibility is explicitly out of scope. A target below roughly 25 points earns a
warning that no budget may reach it, because that is the shape of F14, but the real
answer comes from the engine's `attainable_frontier` on the actual network. This
module has no network object and does not pretend otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from app import adapters, presenters

# --------------------------------------------------------------------------- #
# Field vocabulary
# --------------------------------------------------------------------------- #

#: Human labels, copied from the sidebar widgets rather than invented here, so a
#: proposal names fields the way the screen the user is looking at names them.
LABELS: dict[str, str] = {
    "mode": "Model",
    "budget": "Capital budget",
    "target_risk_pts": "Target risk (%)",
    "exponent": "Diminishing-returns exponent",
    "value_per_risk_point": "Value per risk point (annual)",
    "iterations": "Iterations",
    "seed": "Seed",
    "run_simulation": "Run Monte Carlo",
    "show_sweep": "Compute budget sensitivity",
}

#: Widget bounds, restated here because a proposal is reviewed before the widget
#: ever sees it. A value the sidebar would reject silently is worse than a clamp the
#: user was told about: `st.number_input` coerces out-of-range state without comment.
EXPONENT_RANGE = (0.1, 1.0)
ITERATIONS_RANGE = (1_000, 200_000)
PERCENT_RANGE = (0.0, 100.0)

#: Below this the target is aggressive enough that the portfolio, not the budget, is
#: the likely binding constraint. The acquired system shipped 20.0 as its default
#: (F14) and reported the solver's `Infeasible` beside a 1e9 budget, which read as a
#: funding shortfall to every one of its users.
AGGRESSIVE_TARGET_PTS = 25.0

FRONTIER_NOTE = (
    "Whether any budget reaches it is decided by the engine's attainable frontier "
    "against the actual intervention portfolio, which is not consulted here."
)


# --------------------------------------------------------------------------- #
# The proposal
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Change:
    """One field, its value now, its proposed value, and why.

    `rationale` is not decoration. It is the only artefact that ties a number on
    screen back to the sentence that moved it, which is precisely what the acquired
    system's direct session-state writes destroyed.
    """

    field: str
    label: str
    current: object
    proposed: object
    rationale: str

    def describe(self) -> str:
        return (
            f"{self.label}: {_render(self.field, self.current)} → "
            f"{_render(self.field, self.proposed)} ({self.rationale})"
        )


@dataclass(frozen=True, slots=True)
class Proposal:
    """What the copilot understood, what it did not, and what it wants to change.

    `unresolved` carries as much weight as `changes`. A proposal that changes one
    field and admits it ignored half the sentence is honest; one that quietly drops
    the half it could not parse is the acquired behaviour with better formatting.
    """

    instruction: str
    changes: tuple[Change, ...] = ()
    warnings: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when nothing would change.

        Deliberately keyed on `changes` alone. A no-op instruction still carries a
        warning saying so, and an unparsed one still carries its `unresolved` entry;
        neither is a reason to offer the user an "Apply" button.
        """
        return not self.changes

    def summary(self) -> str:
        lines: list[str] = []
        if self.changes:
            lines.append(f'Proposed changes for: "{self.instruction.strip()}"')
            lines.extend(f"  • {change.describe()}" for change in self.changes)
        else:
            lines.append(f'No changes proposed for: "{self.instruction.strip()}"')
        if self.warnings:
            lines.append("Before you accept:")
            lines.extend(f"  • {text}" for text in self.warnings)
        if self.unresolved:
            lines.append("Not understood, and therefore not acted on:")
            lines.extend(f"  • {text}" for text in self.unresolved)
        return "\n".join(lines)


class StaleProposal(ValueError):
    """Raised when a proposal is applied to inputs that have since moved.

    A proposal is a record of a decision made against a specific snapshot. Applying
    it to a different snapshot writes a value the human never reviewed, which is the
    audit hole this module exists to close, so it fails loudly instead.
    """


# --------------------------------------------------------------------------- #
# Value parsing
# --------------------------------------------------------------------------- #

#: The sign is part of the number. "set the budget to -100" parsed as a hundred is a
#: sentence read as its own opposite, and the clamp warning is the only place a user
#: finds out what was actually understood.
_NUMBER = r"(-?\d[\d,]*(?:\.\d+)?|-?\.\d+)"

#: `mm` before `m`, `bn` before `b`, longest-first, or "1.2mm" parses as 1.2 million
#: followed by a stray "m".
_SCALES: dict[str, float] = {
    "k": 1e3,
    "thousand": 1e3,
    "mm": 1e6,
    "m": 1e6,
    "million": 1e6,
    "bn": 1e9,
    "b": 1e9,
    "billion": 1e9,
}

_MAGNITUDE_RE = re.compile(
    rf"\$?\s*{_NUMBER}\s*(mm|m|k|bn|b|thousand|million|billion)?(?![a-zA-Z])",
    re.IGNORECASE,
)
#: The `%` sign carries no trailing word boundary of its own — `\b` after a
#: non-word character demands a word character next, so "by 20%" at the end of a
#: sentence would not match and the 20 would be read as twenty dollars.
_PERCENT_RE = re.compile(rf"{_NUMBER}\s*(?:%|percent\b|pct\b)", re.IGNORECASE)


def parse_magnitude(text: str) -> float | None:
    """First number in `text`, with `k`/`m`/`b` suffixes and thousands separators.

    Handles `$500k`, `500,000`, `1.2m`, `750K`, `2 million`. A bare number is taken
    at face value: `500` is five hundred. The acquired copilot multiplied bare
    numbers under 1000 by a million, which meant the interpretation of a sentence
    changed discontinuously at a threshold no user could see.
    """
    match = _MAGNITUDE_RE.search(text)
    if match is None:
        return None
    value = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").lower()
    return value * _SCALES.get(suffix, 1.0)


def parse_percent(text: str) -> float | None:
    """A percentage, written `40%`, `40 percent` or `40 pct`. Points, not fractions."""
    match = _PERCENT_RE.search(text)
    if match is None:
        return None
    return float(match.group(1).replace(",", ""))


def _render(field: str, value: object) -> str:
    """Format a value the way the rest of the app formats it.

    Formatting lives in `presenters`, so it is borrowed rather than reimplemented —
    two copies of the same rendering is the smaller cousin of F6, where the headline
    figure and the sweep chart were computed by two drifted copies of one sum.
    """
    if value is None:
        return "not set"
    if field == "budget" or field == "value_per_risk_point":
        return presenters.money(float(value))
    if field == "target_risk_pts":
        return presenters.percent(float(value))
    if field == "mode":
        return "Legacy parity" if value == adapters.MODE_LEGACY else "Monetary NPV"
    if field == "iterations" or field == "seed":
        return f"{int(value):,}"
    if isinstance(value, bool):
        return "on" if value else "off"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _same(left: object, right: object) -> bool:
    """Equality that does not report float noise as a change."""
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) is bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-9 * max(1.0, abs(float(left)))
    return left == right


# --------------------------------------------------------------------------- #
# Clause handling
# --------------------------------------------------------------------------- #


@dataclass
class _Draft:
    """Mutable working set for one call to `interpret`."""

    changes: list[Change]
    warnings: list[str]
    unresolved: list[str]

    def propose(
        self,
        field: str,
        current: Mapping[str, object],
        proposed: object,
        rationale: str,
    ) -> bool:
        """Record a change, or record that there is nothing to change.

        A Change whose proposed value equals the current one is noise in a review
        queue: it invites the user to approve a rewrite that does nothing and trains
        them to click through the ones that do something. Returns whether a change
        was recorded, so a rule can keep its consequential warnings for the case
        where there are consequences.
        """
        now = current.get(field)
        if _same(now, proposed):
            self.warnings.append(
                f"{LABELS[field]} is already {_render(field, now)}; nothing to change."
            )
            return False
        self.changes.append(
            Change(
                field=field,
                label=LABELS[field],
                current=now,
                proposed=proposed,
                rationale=rationale,
            )
        )
        return True

    def reject(self, clause: str, reason: str) -> None:
        self.unresolved.append(f"{clause.strip()} — {reason}")


#: Clause separators. The comma is only a separator when it is not a thousands
#: separator, or "set the budget to 500,000" splits into two unparseable halves.
_SPLIT_RE = re.compile(
    r",(?!\d)|;|\band then\b|\bthen\b|\band also\b|\balso\b|\band\b", re.I
)

_INCREASE_WORDS = r"increase|raise|add|bump|grow|expand|up|more"
_DECREASE_WORDS = r"decrease|reduce|cut|lower|shrink|trim|drop|down|less"
_OFF_WORDS = r"\b(off|disable|stop|skip|without|no longer|don't|do not|never|no)\b"
_ON_WORDS = r"\b(on|enable|run|include|do|start|show|compute|turn on)\b"


def _split_clauses(instruction: str) -> list[str]:
    parts = _SPLIT_RE.split(instruction)
    return [part.strip() for part in parts if part and part.strip()]


def _wants_off(clause: str) -> bool | None:
    """Whether a toggle clause reads as off, on, or neither."""
    if re.search(_OFF_WORDS, clause, re.I):
        return True
    if re.search(_ON_WORDS, clause, re.I):
        return False
    return None


# --- individual rules ------------------------------------------------------ #
#
# Each rule takes the clause and the snapshot, writes into the draft, and returns
# True if it recognised the clause. Order matters and is set in `_RULES`: the more
# specific trigger has to be offered the clause first, because "set the Monte Carlo
# iterations to 50,000" is an iterations instruction, not a simulation toggle.


def _rule_question(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    """Questions are not instructions, and this module does not answer them.

    The acquired copilot answered "what is my risk?" by reading whatever local
    variable was in scope at that point in the script, which is how a stale
    pre-solve figure got quoted back as the current one. Answering questions is the
    job of the screen, which shows engine output.
    """
    if not clause.strip().endswith("?") and not re.match(
        r"^\s*(what|what's|how|why|which|who|when|where|is|are|do|does|can|should)\b",
        clause,
        re.I,
    ):
        return False
    draft.reject(
        clause,
        "this proposes changes to inputs; it does not answer questions or read "
        "results. The metrics on screen come from the engine",
    )
    return True


def _rule_baseline(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    """Baseline risk is evidence, not a knob, and is refused explicitly.

    It is the one input every risk figure and every target is quoted against, and
    the engine evaluates targets against it after the macro multiplier (F18). Moving
    it from a chat line would restate the whole model's frame of reference in a way
    nobody would notice on the next screen.
    """
    if not re.search(r"\bbaseline\b", clause, re.I):
        return False
    draft.reject(
        clause,
        "the baseline risk is not proposed from here: every risk figure and every "
        "target is quoted against it, so it is changed in the sidebar where it is "
        "visible",
    )
    return True


def _rule_weight(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    """The old `val_opt_weight` has no equivalent to propose.

    Legacy parity's objective weight is fixed at the acquired system's 0.5 because
    reproducing its arithmetic is the only reason that mode exists; monetary mode
    prices risk and lead time instead of weighting them. Accepting "set the weight
    to 0.8" would have to invent one of those.
    """
    if not re.search(r"\b(weight|weighting|priority|prioriti[sz]e)\b", clause, re.I):
        return False
    draft.reject(
        clause,
        "there is no optimisation weight to set. Monetary mode prices risk and lead "
        "time in currency instead of trading them off by weight, and legacy parity's "
        "weight is pinned to the acquired system's value so its numbers reproduce",
    )
    return True


def _rule_mode(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    if re.search(r"\blegacy\b|\bparity\b", clause, re.I):
        proposed = adapters.MODE_LEGACY
        rationale = "you asked for legacy parity"
    elif re.search(r"\bmonetary\b|\bnpv\b", clause, re.I):
        proposed = adapters.MODE_MONETARY
        rationale = "you asked for the monetary NPV model"
    else:
        return False

    changed = draft.propose("mode", current, proposed, rationale)
    if changed and proposed == adapters.MODE_LEGACY:
        draft.warnings.append(
            "Legacy parity reproduces the acquired system's arithmetic including its "
            "known defects: it mixes percentage points with days in one objective and "
            "disables the risk cap. Figures are reproducible, not defensible."
        )
        if current.get("target_risk_pts") is not None:
            draft.warnings.append(
                "A risk target is set and legacy parity runs without the risk cap, so "
                "reported risk in that mode is not clipped at the baseline."
            )
    return True


def _rule_target_off(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    triggers = re.search(
        r"(off|disable|clear|remove|cancel|stop|forget|drop)\b[^.]{0,30}\b(target|goal)"
        r"|\b(target|goal)\b[^.]{0,30}\b(off|mode off)"
        r"|spend the (whole |full )?budget instead",
        clause,
        re.I,
    )
    if not triggers:
        return False
    changed = draft.propose(
        "target_risk_pts",
        current,
        None,
        "turning off target mode leaves the objective as buying the most value the "
        "budget allows",
    )
    if changed and current.get("budget") is None:
        draft.warnings.append(
            "No capital budget is set. Outside target mode the budget is the binding "
            "constraint, so it needs a value before the model means anything."
        )
    return True


def _rule_target_on(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    """Enabling target mode, with the target stated.

    Never enabled without a number. The acquired copilot set `val_target_mode=True`
    for any sentence containing "risk" and any digit, and left the target at
    whatever `val_target_risk_goal` happened to hold — its shipped 20.0, which no
    portfolio in the default network can reach at any budget (F14).
    """
    if re.search(r"per risk point|value per|price of risk", clause, re.I):
        return False
    mentions_target = re.search(
        r"\btarget\b|\bgoal\b|\bhit\b|\breach\b|\bunder\b|\bbelow\b|\baim\b|"
        r"\bget\b|\bachieve\b|\bdown to\b|\bat most\b",
        clause,
        re.I,
    )
    if not (re.search(r"\brisk\b", clause, re.I) and mentions_target):
        return False

    value = parse_percent(clause)
    if value is None:
        value = parse_magnitude(clause)
    if value is None:
        draft.reject(
            clause,
            "target mode is never enabled without a stated target, and no number was "
            "given",
        )
        return True

    target, clamped = _clamp(value, *PERCENT_RANGE)
    if clamped:
        draft.warnings.append(
            f"A risk target of {value:g} is outside 0–100%, so {target:g}% is "
            "proposed instead."
        )

    enabling = current.get("target_risk_pts") is None
    changed = draft.propose(
        "target_risk_pts",
        current,
        target,
        f"you asked for risk at or below {target:g}%, which is target mode: the "
        "cheapest portfolio that reaches the target",
    )
    if not changed:
        return True

    if target < AGGRESSIVE_TARGET_PTS:
        draft.warnings.append(
            f"A {target:g}% target is aggressive enough that the portfolio, rather "
            "than the budget, is the likely binding constraint — the acquired "
            f"system shipped a 20% default that no budget could reach. {FRONTIER_NOTE}"
        )
    baseline = current.get("baseline_risk_pts")
    if isinstance(baseline, (int, float)) and target >= float(baseline):
        draft.warnings.append(
            f"A {target:g}% target is at or above the baseline of {float(baseline):g}%, "
            "so it is met by funding nothing. The engine evaluates it against the "
            "macro-adjusted reporting baseline, which is higher still."
        )
    if enabling and current.get("budget") is not None:
        draft.warnings.append(
            "Target mode minimises capital, so the capital budget stops being a "
            "spend goal and becomes a ceiling on what may be spent."
        )
    if enabling and current.get("mode") == adapters.MODE_LEGACY:
        draft.warnings.append(
            "Legacy parity runs without the risk cap, so a target evaluated in that "
            "mode is not comparable with the monetary model's."
        )
    return True


def _rule_iterations(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    if not re.search(r"\biterations?\b|\bdraws\b|\btrials\b", clause, re.I):
        return False
    value = parse_magnitude(clause)
    if value is None:
        draft.reject(clause, "no iteration count was given, and one is not assumed")
        return True
    count, clamped = _clamp(round(value), *ITERATIONS_RANGE)
    count = int(count)
    if clamped:
        draft.warnings.append(
            f"{int(value):,} iterations is outside the supported "
            f"{ITERATIONS_RANGE[0]:,}–{ITERATIONS_RANGE[1]:,} range, so {count:,} is "
            "proposed instead."
        )
    changed = draft.propose(
        "iterations",
        current,
        count,
        f"you asked for {count:,} Monte Carlo iterations",
    )
    if changed and not current.get("run_simulation"):
        draft.warnings.append(
            "The Monte Carlo is currently switched off, so an iteration count has no "
            "effect until it is switched back on."
        )
    return True


def _rule_simulation(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    if not re.search(r"monte ?carlo|\bsimulat(e|ion|ions)\b|\bmc\b", clause, re.I):
        return False
    off = _wants_off(clause)
    if off is None:
        draft.reject(
            clause, "it is not clear whether the Monte Carlo should be on or off"
        )
        return True
    changed = draft.propose(
        "run_simulation",
        current,
        not off,
        "you asked for the Monte Carlo to be " + ("off" if off else "on"),
    )
    if changed and off:
        draft.warnings.append(
            "Without the simulation there are no P50 or P90 figures, so the plan is "
            "reported on its deterministic outcome only."
        )
    return True


def _rule_sweep(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    if not re.search(r"\bsweep\b|budget sensitivity|\bsensitivit(y|ies)\b", clause, re.I):
        return False
    off = _wants_off(clause)
    if off is None:
        draft.reject(
            clause, "it is not clear whether the budget sweep should be on or off"
        )
        return True
    changed = draft.propose(
        "show_sweep",
        current,
        not off,
        "you asked for the budget sensitivity sweep to be " + ("off" if off else "on"),
    )
    if changed and off:
        draft.warnings.append(
            "Without the sweep there is nothing on screen distinguishing a budget "
            "that is binding from one that is already saturated."
        )
    return True


def _rule_exponent(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    if not re.search(r"\bexponent\b|diminishing returns?", clause, re.I):
        return False
    value = parse_magnitude(clause)
    if value is None:
        draft.reject(clause, "no exponent was given")
        return True
    exponent, clamped = _clamp(value, *EXPONENT_RANGE)
    if clamped:
        draft.warnings.append(
            f"An exponent of {value:g} is outside the supported "
            f"{EXPONENT_RANGE[0]:g}–{EXPONENT_RANGE[1]:g} range, so {exponent:g} is "
            "proposed instead. Note that the exponent is a power, not a percentage."
        )
    if not draft.propose(
        "exponent",
        current,
        exponent,
        f"you asked for a diminishing-returns exponent of {exponent:g}",
    ):
        return True
    draft.warnings.append(
        "The exponent shapes how funding converts into risk reduction and has no "
        "stated empirical basis; it is inherited from the acquired system. At 1.0 "
        "the response is linear in funding."
    )
    return True


def _rule_value_per_point(
    clause: str, current: Mapping[str, object], draft: _Draft
) -> bool:
    if not re.search(
        r"per risk point|value of a risk point|price of a risk point", clause, re.I
    ):
        return False
    value = parse_magnitude(clause)
    if value is None:
        draft.reject(clause, "no value per risk point was given")
        return True
    price, clamped = _clamp(value, 0.0, None)
    if clamped:
        draft.warnings.append(
            "A negative value per risk point is meaningless; 0 is proposed."
        )
    if not draft.propose(
        "value_per_risk_point",
        current,
        float(price),
        f"you asked to price a risk point at {presenters.money(float(price))} a year",
    ):
        return True
    draft.warnings.append(
        "Every currency figure in the app is only as defensible as this price. "
        "Changing it from an instruction leaves it unsourced unless the sidebar's "
        "provenance field is updated too."
    )
    return True


def _rule_seed(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    if not re.search(r"\bseed\b", clause, re.I):
        return False
    value = parse_magnitude(clause)
    if value is None:
        draft.reject(clause, "no seed was given")
        return True
    seed, clamped = _clamp(round(value), 0, None)
    if clamped:
        draft.warnings.append("A negative seed is not accepted; 0 is proposed.")
    draft.propose("seed", current, int(seed), f"you asked for seed {int(seed)}")
    return True


def _rule_budget(clause: str, current: Mapping[str, object], draft: _Draft) -> bool:
    """Absolute and relative capital budget instructions.

    Relative forms resolve against the snapshot, so they are refused outright when
    there is no budget to resolve against — in target mode without a spend cap
    `budget` is `None`, and "increase it by 20%" has no meaning there. The acquired
    system would have read the bare 20 as a budget of twenty million dollars.
    """
    if not re.search(
        r"\bbudget\b|\bcapital\b|\bspend(ing)?\b|\bcap\b|\bfunding\b", clause, re.I
    ):
        return False

    now = current.get("budget")
    proposed: float | None = None
    rationale = ""

    factor_match = re.search(
        r"\b(double|doubling|twice|triple|tripling|halve|half)\b", clause, re.I
    )
    percent = parse_percent(clause)
    # "by" marks a relative change, and so do the arithmetic verbs on their own:
    # "add 250,000 to the budget" is a relative instruction that happens to contain
    # the word "to". "increase the budget to 1m" is not, and reading it as one would
    # authorise 1.75m.
    relative = (
        bool(re.search(r"\bby\b|\b(add|subtract|plus|minus|take off)\b", clause, re.I))
        or factor_match is not None
    )
    direction = 0
    if re.search(rf"\b({_INCREASE_WORDS})\b", clause, re.I):
        direction = 1
    elif re.search(rf"\b({_DECREASE_WORDS})\b", clause, re.I):
        direction = -1

    if relative and not isinstance(now, (int, float)):
        draft.reject(
            clause,
            "a relative change needs a budget to be relative to, and none is set. "
            "State the figure you want instead",
        )
        return True

    if factor_match is not None:
        word = factor_match.group(1).lower()
        factors = {"halve": 0.5, "half": 0.5, "triple": 3.0, "tripling": 3.0}
        factor = factors.get(word, 2.0)
        proposed = float(now) * factor
        rationale = f"{word} the current {presenters.money(float(now))}"
    elif relative and percent is not None:
        if direction == 0:
            draft.reject(
                clause,
                f"a change of {percent:g}% was asked for without saying up or down",
            )
            return True
        proposed = float(now) * (1.0 + direction * percent / 100.0)
        rationale = (
            f"{percent:g}% {'above' if direction > 0 else 'below'} the current "
            f"{presenters.money(float(now))}"
        )
    elif relative:
        amount = parse_magnitude(clause)
        if amount is None:
            draft.reject(clause, "no amount was given for the change")
            return True
        if direction == 0:
            draft.reject(
                clause,
                f"a change of {presenters.money(amount)} was asked for without saying "
                "up or down",
            )
            return True
        proposed = float(now) + direction * amount
        rationale = (
            f"{presenters.money(amount)} {'more than' if direction > 0 else 'less than'} "
            f"the current {presenters.money(float(now))}"
        )
    else:
        amount = parse_magnitude(clause)
        if amount is None:
            draft.reject(
                clause,
                "no figure was given, so no budget is assumed",
            )
            return True
        proposed = amount
        rationale = f"you asked for a capital budget of {presenters.money(amount)}"

    budget, clamped = _clamp(proposed, 0.0, None)
    if clamped:
        draft.warnings.append(
            f"That works out to {presenters.money(proposed)}, which is not a budget; "
            f"{presenters.money(float(budget))} is proposed instead."
        )
    if not draft.propose("budget", current, float(budget), rationale):
        return True
    if float(budget) == 0.0:
        draft.warnings.append(
            "A zero budget funds nothing. The result will be an empty portfolio, not "
            "a recommendation to do nothing."
        )
    if current.get("target_risk_pts") is not None:
        draft.warnings.append(
            "Target mode is on, so this figure caps what may be spent to reach the "
            "target rather than setting what will be spent. It does not switch target "
            "mode off; ask for that explicitly if that is what you meant."
        )
    return True


_Rule = Callable[[str, Mapping[str, object], _Draft], bool]

#: Ordered. `_rule_question` and the two refusals go first so a question about the
#: budget is not read as an instruction to change it; `_rule_iterations` precedes
#: `_rule_simulation` and `_rule_value_per_point` precedes `_rule_target_on` because
#: the narrower trigger has to win.
_RULES: tuple[_Rule, ...] = (
    _rule_question,
    _rule_baseline,
    _rule_weight,
    _rule_mode,
    _rule_target_off,
    _rule_iterations,
    _rule_value_per_point,
    _rule_target_on,
    _rule_simulation,
    _rule_sweep,
    _rule_exponent,
    _rule_seed,
    _rule_budget,
)


def _clamp(value: float, low: float | None, high: float | None) -> tuple[float, bool]:
    result = value
    if low is not None and result < low:
        result = low
    if high is not None and result > high:
        result = high
    return result, result != value


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def interpret(instruction: str, current: Mapping[str, object]) -> Proposal:
    """Read an instruction against a snapshot of the inputs and propose changes.

    Deterministic and offline: a regex pass over clauses, resolved against `current`.
    Nothing is solved, nothing is fetched, and nothing outside the returned object is
    touched. `current` is keyed by `Inputs` field names.

    An unrecognised clause becomes an `unresolved` entry and never a change, so a
    sentence this module half-understands produces exactly the changes it did
    understand plus a written admission of the rest.
    """
    draft = _Draft(changes=[], warnings=[], unresolved=[])
    if instruction is None or not instruction.strip():
        return Proposal(instruction=instruction or "")

    for clause in _split_clauses(instruction):
        if not re.search(r"[a-zA-Z0-9]", clause):
            continue
        if not any(rule(clause, current, draft) for rule in _RULES):
            draft.reject(clause, "not recognised as an instruction about any input")

    changes, conflicts = _resolve_conflicts(draft.changes)
    return Proposal(
        instruction=instruction,
        changes=tuple(changes),
        warnings=tuple(conflicts) + tuple(draft.warnings),
        unresolved=tuple(draft.unresolved),
    )


def _resolve_conflicts(changes: Sequence[Change]) -> tuple[list[Change], list[str]]:
    """One field, one proposed value.

    Two clauses touching the same field would otherwise both appear in the review
    list, and applying them in order would leave the user with the value from a line
    they may not have realised was the deciding one. The later clause wins, and the
    fact that it did is stated.
    """
    kept: dict[str, Change] = {}
    warnings: list[str] = []
    for change in changes:
        previous = kept.get(change.field)
        if previous is not None:
            warnings.append(
                f"{change.label} was named twice in one instruction "
                f"({_render(change.field, previous.proposed)} and "
                f"{_render(change.field, change.proposed)}); the later value is "
                "proposed."
            )
        kept[change.field] = change
    return list(kept.values()), warnings


def apply_proposal(current: Mapping[str, object], proposal: Proposal) -> dict:
    """Return a new snapshot with the proposal's changes applied.

    Never mutates `current`: the caller keeps the old snapshot, which is what makes
    an undo and an audit trail possible at all — the thing the acquired copilot's
    direct session-state writes made impossible.

    Raises `StaleProposal` if a field has moved since the proposal was built. A
    reviewed proposal is a decision about specific values; silently applying it over
    a different set writes something nobody approved.
    """
    updated = dict(current)
    for change in proposal.changes:
        if change.field in current and not _same(current[change.field], change.current):
            raise StaleProposal(
                f"{change.label} was {_render(change.field, change.current)} when this "
                f"was proposed and is now "
                f"{_render(change.field, current[change.field])}. "
                "Re-read the instruction against the current inputs."
            )
        updated[change.field] = change.proposed
    return updated
