"""Who is using the app, and what happened when they tried to sign in."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated identity.

    ``user_id`` is what scopes every stored portfolio, so it must be stable across
    sessions and must never be the email address: people change those, and a store
    keyed on a mutable identifier orphans data the moment they do.
    """

    user_id: str
    email: str = ""
    display_name: str = ""
    is_anonymous: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.email or self.user_id

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError("a principal needs a user_id")


#: Used when auth is disabled entirely, so single-user local runs still persist.
#: A fixed id rather than a random one per session, or "persistence" would appear to
#: work and then lose everything on restart.
ANONYMOUS = Principal(
    user_id="local-single-user",
    display_name="Local user",
    is_anonymous=True,
)


@dataclass(frozen=True, slots=True)
class AuthOutcome:
    """The result of an attempt to sign in.

    Deliberately not an exception. A failed sign-in is an ordinary event that the UI
    renders, and raising would invite a bare `except` in the view layer of the sort
    that hid every failure in the acquired system.
    """

    principal: Principal | None = None
    error: str = ""
    locked_out: bool = False
    retry_after_seconds: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def succeeded(self) -> bool:
        return self.principal is not None
