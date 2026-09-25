# praxis/security/api_key.py
"""API key generation and verification.

Only the SHA-256 hash of a key is ever persisted (`ApiKey.key_hash`).
The plaintext is returned exactly once, from `generate_api_key`, and
never logged - `praxis.security.redaction` also catches the
`praxis_sk_` prefix defensively in case one is ever passed into a log
payload by mistake.

SHA-256 (not a slow KDF like bcrypt/argon2) is the right choice here
and this is deliberate, not a shortcut: these are 256-bit
machine-generated random tokens, not human-chosen passwords, so there
is no dictionary/brute-force surface for a slow hash to defend against,
and key verification sits on the hot path of every single request.
"""
from __future__ import annotations

import hashlib
import secrets

# A recognizable prefix so a leaked key is greppable in logs/repos and
# so `redaction.py` can pattern-match it, mirroring how every provider
# in this codebase's own detectors prefixes theirs (sk-ant-, xoxb-, ...).
KEY_PREFIX = "praxis_sk_"

# 32 bytes -> 43 url-safe base64 characters. Well past any brute-force
# concern while staying a single copy-pasteable token.
_KEY_BYTES = 32


def generate_api_key() -> tuple[str, str]:
    """Returns `(plaintext_key, key_hash)`.

    The caller stores only the hash and shows the plaintext to the
    operator exactly once.
    """
    plaintext = f"{KEY_PREFIX}{secrets.token_urlsafe(_KEY_BYTES)}"
    return plaintext, hash_api_key(plaintext)


def hash_api_key(plaintext: str) -> str:
    """The stored representation of `plaintext`."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def looks_like_api_key(value: str) -> bool:
    """Cheap shape check used to give a clearer 401 ("malformed key")
    than a bare lookup miss would."""
    return value.startswith(KEY_PREFIX) and len(value) > len(KEY_PREFIX) + 20

