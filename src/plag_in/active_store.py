"""A receipt writer that follows the active-store pointer while it runs.

A gateway resolves its receipt store when the serving context is built. That
made the pointer a start-up parameter rather than the record of which store is
active: after `plag-in receipt --initialise-current-store` activated a
current-schema store, a gateway already running stayed bound to the store it
had opened, and either kept appending to the legacy store or refused every
write there until it was restarted. No restart requirement was documented,
because none was intended (independent review of work package A, second pass,
D2).

This class holds the binding instead. Every operation resolves the pointer,
and the underlying store is rebuilt when, and only when, the resolved paths
change. The pointer read is one `open` and one small `read` of a file in the
state directory, on a path that is already touched once per receipt.

Two properties of the design are deliberate and are limits, not oversights.

Rebinding is per operation, so an activation between two appends is honoured
between them and never inside one. An append that has begun completes against
the store it resolved, under that store's lock and checkpoint.

`get()` reads the active store only. A receipt written to the legacy store
before activation is no longer served by its request ID afterwards. That is
already true across a restart, and the alternative is to search a store the
operator has moved off, which is the silent fallback `read_receipt_pointer`
exists to refuse. The receipt is still in the legacy store, which `plag-in
status` continues to report by path.
"""
from __future__ import annotations

import threading
from pathlib import Path

from plag_in.paths import ActiveStoreSnapshot, StateLayout
from plag_in.receipts import Receipt, ReceiptStore


class ActiveReceiptStore:
    """`ReceiptStore`'s writing interface, bound to whichever store is active.

    Substitutable for a `ReceiptStore` in `GatewayContext`: it carries the
    `append`, `get`, `verify_chain` and `require_valid_chain` operations and
    the `path` and `hmac_key_path` attributes that callers read.
    """

    def __init__(self, state: StateLayout):
        self._state = state
        self._lock = threading.Lock()
        self._bound: ReceiptStore | None = None
        self._snapshot: ActiveStoreSnapshot | None = None
        self._rebind_count = 0
        # Resolve once here so an unreadable or escaping pointer fails when the
        # gateway is being built, not on the first request after it is serving.
        self.resolve()

    def resolve(self) -> ReceiptStore:
        """Return the store the pointer names now, rebinding if it has changed.

        The snapshot is taken outside the lock: reading the pointer is what
        this call is for, and holding the lock across it would serialise every
        append behind one file read for no gain. The lock covers the compare
        and the swap only.
        """
        snapshot = self._state.active_store_snapshot()
        with self._lock:
            if (
                self._bound is None
                or self._snapshot.store != snapshot.store
                or self._snapshot.hmac_key != snapshot.hmac_key
            ):
                self._bound = ReceiptStore(snapshot.store, hmac_key_path=snapshot.hmac_key)
                self._snapshot = snapshot
                self._rebind_count += 1
            return self._bound

    # -- reporting -----------------------------------------------------
    #
    # Not part of the ReceiptStore interface. Both exist so a test can state
    # what happened rather than infer it from where a record landed.

    @property
    def rebind_count(self) -> int:
        """How many times a different store has been bound, first bind included."""
        return self._rebind_count

    @property
    def snapshot(self) -> ActiveStoreSnapshot:
        """The snapshot the current binding was built from."""
        self.resolve()
        return self._snapshot

    # -- ReceiptStore interface ----------------------------------------

    @property
    def path(self) -> Path:
        return self.resolve().path

    @property
    def hmac_key_path(self) -> Path:
        return self.resolve().hmac_key_path

    def append(self, receipt: Receipt) -> dict:
        return self.resolve().append(receipt)

    def get(self, request_id: str) -> dict:
        return self.resolve().get(request_id)

    def verify_chain(self) -> tuple[bool, int]:
        return self.resolve().verify_chain()

    def require_valid_chain(self) -> int:
        return self.resolve().require_valid_chain()
