"""Sign-in, with the lockout somewhere refreshing the page cannot reach.

The acquired system's auth had three problems worth naming, because the shape of this
module is a response to each.

**The lockout was in session state.** It counted five failed attempts in
`st.session_state` and then refused to try again. Pressing F5 reset the counter, so the
control communicated diligence without providing any. Here the attempts live in the
store, keyed by email, so the limit survives a refresh, a new tab, and a restart.

**Failures were indistinguishable.** A wrong password, an unreachable auth service and
a misconfigured key all produced the same red message. The three call for completely
different actions, so `AuthOutcome` separates them.

**Passwords were compared directly.** This module never stores or compares a plaintext
password: `LocalPasswordProvider` uses PBKDF2-HMAC-SHA256 with a per-user salt and a
constant-time comparison.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Protocol

from app.auth.models import ANONYMOUS, AuthOutcome, Principal

#: Failed attempts allowed inside the window before the account is held.
MAX_FAILURES = 5
#: How long the window is. Not a permanent lock: a permanent one turns a forgotten
#: password into a support ticket and hands anyone a trivial denial-of-service.
LOCKOUT_WINDOW = timedelta(minutes=15)

PBKDF2_ITERATIONS = 240_000


class AuthProvider(Protocol):
    """Anything that can turn credentials into a principal."""

    name: str

    def sign_in(self, email: str, password: str) -> AuthOutcome: ...

    @property
    def is_configured(self) -> bool: ...


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Return ``pbkdf2_sha256$iterations$salt$hash``."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time verification.

    `hmac.compare_digest` rather than `==`: string comparison short-circuits on the
    first differing byte, which leaks how much of a guess was correct.
    """
    try:
        algorithm, iterations, salt_hex, expected_hex = encoded.split("$")
    except (ValueError, AttributeError):
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    try:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), expected_hex)


# --------------------------------------------------------------------------- #
# Lockout
# --------------------------------------------------------------------------- #


class LockoutPolicy:
    """Rate-limits sign-in attempts using a durable store."""

    def __init__(self, store, *, max_failures: int = MAX_FAILURES, window: timedelta = LOCKOUT_WINDOW):
        self._store = store
        self.max_failures = max_failures
        self.window = window

    def check(self, email: str) -> AuthOutcome | None:
        """Return a locked-out outcome, or ``None`` to proceed."""
        if self._store is None:
            return None
        since = datetime.now(UTC) - self.window
        failures = self._store.recent_failures(email, since=since)
        if failures < self.max_failures:
            return None
        return AuthOutcome(
            error=(
                f"Too many failed sign-in attempts. Try again in "
                f"{int(self.window.total_seconds() // 60)} minutes."
            ),
            locked_out=True,
            retry_after_seconds=int(self.window.total_seconds()),
        )

    def record(self, email: str, *, succeeded: bool) -> None:
        if self._store is None:
            return
        self._store.record_attempt(email, succeeded=succeeded)
        if succeeded:
            # Otherwise a user who fails four times and then succeeds stays one
            # mistake away from being locked out of an account they just proved they
            # own.
            self._store.clear_attempts(email)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


class AnonymousProvider:
    """No authentication. Every session is the same local user.

    The right default for a single-user local run: it keeps persistence working without
    demanding credentials, and because the principal id is fixed rather than random,
    saved portfolios survive a restart.
    """

    name = "anonymous"
    is_configured = True

    def sign_in(self, email: str = "", password: str = "") -> AuthOutcome:
        return AuthOutcome(principal=ANONYMOUS)


class LocalPasswordProvider:
    """Users from an environment variable, for development and self-hosting.

    `SCRCAE_USERS` holds JSON: ``{"someone@example.com": "<encoded hash>"}``, where the
    hash comes from `hash_password`. Plaintext passwords are rejected rather than
    accepted with a warning, because a development convenience that accepts plaintext
    is the thing that ends up in production.
    """

    name = "local"

    def __init__(self, users: dict[str, str] | None = None, *, lockout: LockoutPolicy | None = None):
        self._users = {k.strip().lower(): v for k, v in (users or self._from_env()).items()}
        self._lockout = lockout

    @staticmethod
    def _from_env() -> dict[str, str]:
        raw = os.environ.get("SCRCAE_USERS", "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @property
    def is_configured(self) -> bool:
        return bool(self._users)

    def sign_in(self, email: str, password: str) -> AuthOutcome:
        address = (email or "").strip().lower()
        if not address or not password:
            return AuthOutcome(error="Enter an email address and a password.")

        if self._lockout is not None:
            held = self._lockout.check(address)
            if held is not None:
                return held

        encoded = self._users.get(address)
        if encoded and not encoded.startswith("pbkdf2_sha256$"):
            return AuthOutcome(
                error=(
                    "This account is configured with a plaintext password, which is "
                    "not accepted. Generate a hash with "
                    "`python -m app.auth.hash_password`."
                )
            )

        # Hash even when the account is unknown, so a missing account and a wrong
        # password take the same time and neither can be identified by timing.
        reference = encoded or hash_password(secrets.token_hex(16))
        ok = verify_password(password, reference) and encoded is not None

        if self._lockout is not None:
            self._lockout.record(address, succeeded=ok)

        if not ok:
            # One message for both cases: naming which was wrong confirms that an
            # account exists.
            return AuthOutcome(error="That email address and password do not match.")

        return AuthOutcome(
            principal=Principal(
                user_id=f"local:{address}",
                email=address,
                display_name=address.split("@")[0],
            )
        )


class SupabasePasswordProvider:
    """Supabase email/password auth.

    `supabase` is imported inside the method rather than at module scope so the app
    runs, and the tests pass, without the package installed. The acquired system
    imported it at the top of the file, which made the whole application unstartable
    when the dependency or its credentials were absent.
    """

    name = "supabase"

    def __init__(self, url: str = "", key: str = "", *, lockout: LockoutPolicy | None = None):
        self.url = url or os.environ.get("SUPABASE_URL", "").strip()
        self.key = key or os.environ.get("SUPABASE_ANON_KEY", "").strip()
        self._lockout = lockout

    @property
    def is_configured(self) -> bool:
        return bool(self.url and self.key)

    def sign_in(self, email: str, password: str) -> AuthOutcome:
        address = (email or "").strip().lower()
        if not self.is_configured:
            return AuthOutcome(
                error=(
                    "Supabase is not configured. Set SUPABASE_URL and "
                    "SUPABASE_ANON_KEY, or run with local accounts."
                )
            )
        if not address or not password:
            return AuthOutcome(error="Enter an email address and a password.")

        if self._lockout is not None:
            held = self._lockout.check(address)
            if held is not None:
                return held

        try:
            from supabase import create_client
        except ImportError:
            return AuthOutcome(
                error="The supabase package is not installed. Run `pip install supabase`."
            )

        try:
            client = create_client(self.url, self.key)
            response = client.auth.sign_in_with_password(
                {"email": address, "password": password}
            )
        except Exception as exc:  # noqa: BLE001
            # Reported as a service failure, not as bad credentials. Telling a user
            # their password is wrong when the auth service is unreachable sends them
            # to reset a password that was fine.
            if self._lockout is not None:
                self._lockout.record(address, succeeded=False)
            return AuthOutcome(
                error=f"Could not reach the authentication service ({type(exc).__name__})."
            )

        user = getattr(response, "user", None)
        if user is None:
            if self._lockout is not None:
                self._lockout.record(address, succeeded=False)
            return AuthOutcome(error="That email address and password do not match.")

        if self._lockout is not None:
            self._lockout.record(address, succeeded=True)
        return AuthOutcome(
            principal=Principal(
                user_id=f"supabase:{user.id}",
                email=getattr(user, "email", address) or address,
                display_name=(getattr(user, "email", address) or address).split("@")[0],
            )
        )


def build_provider(store=None) -> AuthProvider:
    """Pick a provider from the environment, preferring the strongest configured one.

    Falls back to anonymous rather than to a locked door, and the UI states which
    provider is active. A single-user local run should not need credentials, but it
    must never be ambiguous whether anyone was actually authenticated.
    """
    lockout = LockoutPolicy(store)
    supabase = SupabasePasswordProvider(lockout=lockout)
    if supabase.is_configured:
        return supabase
    local = LocalPasswordProvider(lockout=lockout)
    if local.is_configured:
        return local
    return AnonymousProvider()
