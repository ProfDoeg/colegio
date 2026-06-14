"""Tag classifier — strand vs tag (intact / SPENT), and parity with the monolith.

Hermetic: a synthetic root with one strand output, one intact tag, one spent
tag, driven by mocked spend_of/get_tx. Plus a differential proving
colegio.tags.classify_root_outputs == quipu_tags.classify_root_outputs.
"""

import os
import sys

import pytest

import pydoge
from colegio import tags

ROOT = pydoge.serialize(pydoge.mktx(
    ["aa" * 32 + ":0"],
    [{"value": 1000, "script": "76a914" + "00" * 20 + "88ac"},   # vout0 — strand
     {"value": 2000, "script": "76a914" + "11" * 20 + "88ac"},   # vout1 — intact tag
     {"value": 3000, "script": "76a914" + "22" * 20 + "88ac"}],  # vout2 — spent tag
))
ROOT_TXID = pydoge.txhash(ROOT)
STRAND_SPENDER = "11" * 32        # its tx carries an OP_RETURN -> strand
EVENT_SPENDER = "22" * 32         # its tx has no OP_RETURN -> tag spent by an event


def _spend_of(txid, vout):
    return {(ROOT_TXID, 0): STRAND_SPENDER,
            (ROOT_TXID, 2): EVENT_SPENDER}.get((txid, vout))   # vout1 unspent -> None


def _get_tx(spender):
    if spender == STRAND_SPENDER:
        return {"outs": [{"script": "6a04deadbeef"},
                         {"script": "76a914" + "44" * 20 + "88ac"}]}
    return {"outs": [{"script": "76a914" + "33" * 20 + "88ac"}]}   # no OP_RETURN


def test_classify_strand_intact_spent():
    res = tags.classify_root_outputs(ROOT, _spend_of, _get_tx)
    assert [c["kind"] for c in res] == ["strand", "tag", "tag"]
    assert [c["state"] for c in res] == ["strand", "intact", "SPENT"]
    assert [c["spent_by"] for c in res] == [STRAND_SPENDER, None, EVENT_SPENDER]
    assert [c["value"] for c in res] == [1000, 2000, 3000]
    assert tags.find_tags(ROOT, _spend_of, _get_tx) == [res[1], res[2]]


def test_accepts_dict_root_equally():
    from_hex = tags.classify_root_outputs(ROOT, _spend_of, _get_tx)
    from_dict = tags.classify_root_outputs(pydoge.deserialize(ROOT), _spend_of, _get_tx)
    assert from_hex == from_dict


def test_parity_with_monolith_quipu_tags():
    cinv = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "..", "Colegio_Invisible"))
    if not os.path.isdir(cinv):
        pytest.skip("Colegio_Invisible not present")
    if cinv not in sys.path:
        sys.path.insert(0, cinv)
    try:
        import quipu_tags as old
    except Exception as exc:
        pytest.skip(f"quipu_tags not importable: {exc}")
    assert (tags.classify_root_outputs(ROOT, _spend_of, _get_tx)
            == old.classify_root_outputs(ROOT, _spend_of, _get_tx))
