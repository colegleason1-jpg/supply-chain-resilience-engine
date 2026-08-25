"""Generate a password hash for `SCRCAE_USERS`.

    python -m app.auth.hash_password

Reads the password without echoing it and prints only the encoded hash, so the
plaintext never reaches a shell history or a terminal scrollback buffer.
"""

from __future__ import annotations

import getpass
import json
import sys

from app.auth.providers import hash_password


def main() -> int:
    email = input("email: ").strip().lower()
    password = getpass.getpass("password: ")
    if not email or not password:
        print("both an email address and a password are required", file=sys.stderr)
        return 1
    confirmation = getpass.getpass("confirm: ")
    if password != confirmation:
        print("passwords do not match", file=sys.stderr)
        return 1

    print("\nAdd this to SCRCAE_USERS (merge with any existing entries):\n")
    print(json.dumps({email: hash_password(password)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
