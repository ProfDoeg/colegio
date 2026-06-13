"""Hermetic tests for the reader's pure logic — OP_RETURN parsing and the
dataframe strand/quipu walkers — differential against the monolith.

The RPC-backed paths (scan_accounts, the RPC branch of read_strand/read_quipu,
fetch_quipu_bytes) need a live node and are exercised separately; here we prove
the decode + walk logic with synthetic frames, asserting the new package agrees
with the original `colegio_tools` byte-for-byte. Skips if the monolith isn't
importable.
"""

import os
import sys

import pandas as pd
import pytest

from colegio import reading

_HERE = os.path.dirname(os.path.abspath(__file__))
_MONO_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "Colegio_Invisible"))
if _MONO_DIR not in sys.path:
    sys.path.insert(0, _MONO_DIR)
try:
    import colegio_tools as old
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"monolith colegio_tools not importable: {exc}",
                allow_module_level=True)


def _vout(type_, hex_):
    return {"scriptPubKey": {"type": type_, "hex": hex_}}


def test_extract_op_return_short_push():
    v = _vout("nulldata", "6a04deadbeef")  # 4-byte payload
    assert reading.extract_op_return(v) == "deadbeef"
    assert reading.extract_op_return(v) == old.extract_op_return(v)


def test_extract_op_return_pushdata1():
    payload = "ab" * 80  # 80 bytes → OP_PUSHDATA1 (0x4c 0x50)
    v = _vout("nulldata", "6a4c50" + payload)
    assert reading.extract_op_return(v) == payload
    assert reading.extract_op_return(v) == old.extract_op_return(v)


def test_extract_op_return_non_nulldata_is_none():
    v = _vout("pubkeyhash", "76a914" + "11" * 20 + "88ac")
    assert reading.extract_op_return(v) is None
    assert reading.extract_op_return(v) == old.extract_op_return(v)


# A synthetic 2-strand quipu graph:
#   root R: out :0 starts the header strand, out :1 starts a body strand
#   header strand: R:0 -spent_by-> K1 (op c1dd0001aa) -> K2 (op bbbb) -> end
#   body strand:   R:1 -spent_by-> B1 (op cccc) -> end
def _df_outputs():
    rows = [
        {"txout": "R:0", "spent_in": "K1", "op_return": None, "txid": "R"},
        {"txout": "R:1", "spent_in": "B1", "op_return": None, "txid": "R"},
        {"txout": "K1:0", "spent_in": "K2", "op_return": "c1dd0001aa", "txid": "K1"},
        {"txout": "K2:0", "spent_in": None, "op_return": "bbbb", "txid": "K2"},
        {"txout": "B1:0", "spent_in": None, "op_return": "cccc", "txid": "B1"},
    ]
    return pd.DataFrame(rows)


def _df_transactions():
    return pd.DataFrame([
        {"txid": "R", "op_return": None,
         "addresses_in_outputs": [["Daddr"], ["Daddr"]]},
        {"txid": "K1", "op_return": "c1dd0001aa", "addresses_in_outputs": [["Daddr"]]},
        {"txid": "K2", "op_return": "bbbb", "addresses_in_outputs": [["Daddr"]]},
        {"txid": "B1", "op_return": "cccc", "addresses_in_outputs": [["Daddr"]]},
    ])


def test_read_strand_dataframe_walk_matches_monolith():
    new = reading.read_strand("R:0", df_outputs=_df_outputs())
    ref = old.read_strand("R:0", df_outputs=_df_outputs())
    assert new == ref == "c1dd0001aabbbb"


def test_read_quipu_dataframe_walk_matches_monolith():
    h_new, b_new = reading.read_quipu("R", _df_outputs())
    h_old, b_old = old.read_quipu("R", _df_outputs())
    assert (h_new, b_new) == (h_old, b_old)
    assert h_new == "c1dd0001aabbbb"  # header = K1 ++ K2 payloads
    assert b_new == "cccc"            # body strand = B1 payload


def test_identify_and_find_quipu_roots_match_monolith():
    new_roots = reading.identify_quipus(_df_transactions(), _df_outputs())
    old_roots = old.identify_quipus(_df_transactions(), _df_outputs())
    assert new_roots == old_roots == ["R"]

    new_found = reading.find_quipu_roots("Daddr", _df_transactions(), _df_outputs())
    old_found = old.find_quipu_roots("Daddr", _df_transactions(), _df_outputs())
    assert new_found == old_found == ["R"]


def test_outputs_walk_index_shape():
    idx = reading.outputs_walk_index(_df_outputs())
    assert idx["txout"]["R:0"] == ("K1", None)
    assert idx["txid_op"]["K1"] == "c1dd0001aa"
