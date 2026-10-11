# Legacy upstream tests

These files target the inherited library examples, older training signatures,
and downloaded reference datasets. They are retained as migration references,
not counted as passing coverage of the current runner.

The active suite is `tests/runner`, `tests/numerics`, and `tests/integration`.
Their default pytest collection no longer imports these old Hydra fixtures.
New runner tests belong in the active directories. The legacy suite needs a
separate migration before it can be enabled again.
