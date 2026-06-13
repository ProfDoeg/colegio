"""Byte-identity gate for the money code.

`inscriptions` is the diamond's foundation — it builds and signs real spends.
The non-negotiable gate: the new pydoge-backed Cadena writers must reproduce the
old `cryptos`-backed monolith **byte-for-byte**. Since `cryptos` is the library
that produced the entire on-chain quipu corpus, byte-identity vs cryptos *is*
byte-identity vs the chain.

This runs a differential test: the old monolith classes (imported from the
`Colegio_Invisible` repo) and the new `colegio.inscriptions` classes are driven
through the identical public API on identical inputs — including real on-chain
body payloads (`data/bodies/*.bin`) — and their signed serializations are
asserted equal. Offline only: make_tx/precompute do no network I/O.

Skips cleanly if `cryptos` or the monolith repo is not present.
"""

import copy
import glob
import os
import sys

import pytest

import pydoge
from colegio import inscriptions

cryptos = pytest.importorskip("cryptos")

# Locate the original monolith (sibling repo) and the on-chain body corpus.
_HERE = os.path.dirname(os.path.abspath(__file__))
_MONO_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "Colegio_Invisible"))
_BODIES_DIR = os.path.join(_MONO_DIR, "data", "bodies")

if _MONO_DIR not in sys.path:
    sys.path.insert(0, _MONO_DIR)
try:
    import colegio_tools as old
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"monolith colegio_tools not importable: {exc}",
                allow_module_level=True)

# Fixed test keys (NOT real). 64-hex → uncompressed in both libraries, matching
# the apocrypha-era on-chain convention the corpus was written under.
PRIV = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
PRIV2 = "11" * 32
PRIV3 = "22" * 32

# A well-formed (fake) funding utxo with plenty of value to cover every tip.
UTXO = {"output": "ab" * 32 + ":0", "value": 1_000_000_000}
TIP = 100_000

_BODY_FILES = sorted(glob.glob(os.path.join(_BODIES_DIR, "*.bin")),
                     key=os.path.getsize)


def _body(min_chunks=1, max_bytes=None):
    """A real on-chain body payload, optionally truncated to bound runtime."""
    if not _BODY_FILES:
        pytest.skip("no on-chain body corpus present")
    raw = open(_BODY_FILES[len(_BODY_FILES) // 2], "rb").read()
    if len(raw) < min_chunks * 80:
        raw = open(_BODY_FILES[-1], "rb").read()
    return raw[:max_bytes] if max_bytes else raw


# --- mk_opreturn primitive (no cryptos needed) ------------------------------

def test_mk_opreturn_short_payload():
    data = b"hello quipu"  # 11 bytes, < 0x4c → bare length prefix
    assert inscriptions.mk_opreturn(data) == "6a0b" + data.hex()
    assert inscriptions.mk_opreturn("hello quipu") == inscriptions.mk_opreturn(data)
    assert inscriptions.mk_opreturn(data, json_out=True) == {
        "script": "6a0b" + data.hex(), "value": 0}


def test_mk_opreturn_80_byte_chunk_uses_pushdata1():
    data = bytes(range(80))  # 0x50 bytes → OP_PUSHDATA1
    assert inscriptions.mk_opreturn(data) == "6a4c50" + data.hex()


def test_mk_opreturn_rawtx_append_matches_monolith():
    addr = pydoge.privtoaddr(PRIV, compressed=False)
    base = pydoge.serialize(pydoge.mktx([UTXO], [{"value": 90000, "address": addr}]))
    new = inscriptions.mk_opreturn(b"payload bytes", rawtx=base)
    ref = old.mk_opreturn(b"payload bytes", rawtx=base)
    assert new == ref
    tx = pydoge.deserialize(new)
    assert len(tx["outs"]) == 2
    assert tx["outs"][-1] == {"value": 0, "script": inscriptions.mk_opreturn(b"payload bytes")}


# --- differential helpers ---------------------------------------------------

# NOTE: cryptos.mktx mutates the input utxo dict in place (rewrites "output"
# → "tx_hash"/"tx_pos"); pydoge.mktx does not. Both are harmless in real flows
# (Cadena.broadcast and precompute's loop reassign head_utxo each step), but the
# differential must hand each instance its OWN copy so neither path pollutes the
# other or the next test.

def _assert_make_tx_identical(NewCls, OldCls, *ctor_args):
    new = NewCls(*copy.deepcopy(ctor_args))
    new.make_tx()
    ref = OldCls(*copy.deepcopy(ctor_args))
    ref.make_tx()
    assert pydoge.serialize(new.signed_inscribed_tx) == \
        cryptos.serialize(ref.signed_inscribed_tx)


def _assert_precompute_identical(NewCls, OldCls, *ctor_args):
    new = NewCls(*copy.deepcopy(ctor_args))
    new.precompute()
    ref = OldCls(*copy.deepcopy(ctor_args))
    ref.precompute()
    assert new.txns == ref.txns        # signed serialized hex, per chunk
    assert new.txn_ids == ref.txn_ids  # threaded txids (head_utxo chaining)
    assert new.state == inscriptions.STATE_PRECOMPUTED


# --- Cadena / CadenaMulti make_tx (byte-identical, real on-chain payload) ----

def test_cadena_make_tx_byte_identical():
    data = _body()  # real on-chain body; make_tx signs chunk 0 (80 B)
    _assert_make_tx_identical(inscriptions.Cadena, old.Cadena, PRIV, data, UTXO, TIP)


def test_cadena_make_tx_short_payload():
    _assert_make_tx_identical(inscriptions.Cadena, old.Cadena,
                              PRIV, b"a short knot", UTXO, TIP)


def test_cadena_multi_make_tx_byte_identical():
    prvs = [PRIV, PRIV2]
    _assert_make_tx_identical(inscriptions.CadenaMulti, old.CadenaMulti,
                              prvs, _body(), UTXO, TIP)


def test_cadena_multi_make_tx_3of3():
    prvs = [PRIV, PRIV2, PRIV3]
    _assert_make_tx_identical(inscriptions.CadenaMulti, old.CadenaMulti,
                              prvs, b"three keys must sign this strand", UTXO, TIP)


def test_cadena_multi_address_matches_monolith():
    prvs = [PRIV, PRIV2]
    new = inscriptions.CadenaMulti(prvs, b"x", dict(UTXO), TIP)
    ref = old.CadenaMulti(prvs, b"x", dict(UTXO), TIP)
    assert new.addr == ref.addr
    assert new.script == ref.script


# --- CadenaAtom / CadenaMultiAtom precompute (threading + byte-identity) -----

def test_cadena_atom_precompute_byte_identical():
    # multi-chunk strand exercises head_utxo→txid threading across the chain
    data = _body(min_chunks=8, max_bytes=800)  # ~10 chunks
    _assert_precompute_identical(inscriptions.CadenaAtom, old.CadenaAtom,
                                 PRIV, data, UTXO, TIP)


def test_cadena_atom_precompute_two_chunk():
    _assert_precompute_identical(inscriptions.CadenaAtom, old.CadenaAtom,
                                 PRIV, _body(max_bytes=120), UTXO, TIP)


def test_cadena_multi_atom_precompute_byte_identical():
    prvs = [PRIV, PRIV2]
    data = _body(min_chunks=8, max_bytes=800)
    _assert_precompute_identical(inscriptions.CadenaMultiAtom, old.CadenaMultiAtom,
                                 prvs, data, UTXO, TIP)


def test_atom_txids_match_serialized_txhash():
    """Each threaded txid must equal pydoge.txhash of its own serialized tx."""
    atom = inscriptions.CadenaAtom(PRIV, _body(max_bytes=400), dict(UTXO), TIP)
    atom.precompute()
    for tx_hex, txid in zip(atom.txns, atom.txn_ids):
        assert pydoge.txhash(tx_hex) == txid
