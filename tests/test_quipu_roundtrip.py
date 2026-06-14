"""End-to-end quipu gate — independent of cryptos AND of the monolith.

The differential tests prove the new code reproduces the old code byte-for-byte.
This file proves the thing that matters in the world: that a *staged* quipu
actually carries its data. It writes with `inscriptions` and reads back with
`reading` — two separate modules, no shared oracle — and asserts the embedded
OP_RETURN payloads reassemble the original bytes, the funding chain threads
correctly, and (when a node is reachable) Dogecoin Core itself decodes the
staged transaction as a standard nulldata OP_RETURN carrying the exact payload.

No funds are ever spent: staging and `decoderawtransaction`/`testmempoolaccept`
are read-only. The node test is opt-in via QUIPU_RPC_TESTS=1 (project convention)
and skips if the node is unreachable.
"""

import glob
import os

import pytest

import pydoge
from colegio import inscriptions, node, reading

PRIV = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
PRIV2 = "11" * 32
UTXO = {"output": "ab" * 32 + ":0", "value": 1_000_000_000}
TIP = 100_000

_HERE = os.path.dirname(os.path.abspath(__file__))
_BODIES_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "..", "Colegio_Invisible", "data", "bodies"))


def _real_body(max_bytes=250):
    """A real on-chain body, sliced so the last chunk is partial (<80 B) — this
    exercises both the OP_PUSHDATA1 path (full 80-B chunks) and the bare-length
    path (the short tail)."""
    files = sorted(glob.glob(os.path.join(_BODIES_DIR, "*.bin")), key=os.path.getsize)
    if not files:
        pytest.skip("no on-chain body corpus present")
    raw = open(files[len(files) // 2], "rb").read()[:max_bytes]
    if len(raw) % 80 == 0:           # guarantee a partial tail chunk
        raw = raw[:-13]
    return raw


def _opreturn_payload_via_reader(out_script_hex):
    """Read an OP_RETURN payload exactly as the reader would off a decoded tx."""
    vout = {"scriptPubKey": {"type": "nulldata", "hex": out_script_hex}}
    return reading.extract_op_return(vout)


def test_atom_strand_embeds_and_reads_back(strand_cls=inscriptions.CadenaAtom,
                                           prv=PRIV):
    """Stage a full single-key strand; read every OP_RETURN back with the reader
    and reassemble the original body. Also checks tx structure + chain threading."""
    body = _real_body()
    chunks = [body[i:i + 80] for i in range(0, len(body), 80)]
    assert len(chunks) >= 3 and len(chunks[-1]) < 80   # multi-chunk, partial tail

    atom = strand_cls(prv, body, dict(UTXO), TIP)
    atom.precompute()
    assert len(atom.txns) == len(chunks)

    reassembled = b""
    prev_outpoint = UTXO["output"]
    prev_value = UTXO["value"]
    for tx_hex, chunk, txid in zip(atom.txns, chunks, atom.txn_ids):
        tx = pydoge.deserialize(tx_hex)

        # structure: one self-pay funding output [0] + one OP_RETURN [1]
        assert len(tx["outs"]) == 2
        assert tx["outs"][0]["value"] == prev_value - TIP
        assert tx["outs"][1]["value"] == 0
        assert tx["outs"][1]["script"].startswith("6a")

        # chain threading: this tx spends the previous tx's vout 0
        spent = f"{tx['ins'][0]['outpoint']['hash']}:{tx['ins'][0]['outpoint']['index']}"
        assert spent == prev_outpoint
        assert pydoge.txhash(tx_hex) == txid

        # the embedded OP_RETURN, read back by the *reader*, is exactly the chunk
        payload_hex = _opreturn_payload_via_reader(tx["outs"][1]["script"])
        assert payload_hex == chunk.hex()
        reassembled += bytes.fromhex(payload_hex)

        prev_outpoint = f"{txid}:0"
        prev_value = prev_value - TIP

    # the whole quipu round-trips: writer → reader → original bytes
    assert reassembled == body


def test_multi_atom_strand_embeds_and_reads_back():
    """Same end-to-end proof for the multisig writer (P2SH funding output)."""
    body = _real_body()
    atom = inscriptions.CadenaMultiAtom([PRIV, PRIV2], body, dict(UTXO), TIP)
    atom.precompute()
    chunks = [body[i:i + 80] for i in range(0, len(body), 80)]

    reassembled = b""
    for tx_hex, chunk in zip(atom.txns, chunks):
        tx = pydoge.deserialize(tx_hex)
        assert len(tx["outs"]) == 2
        # funding output pays the P2SH multisig address (a914..87)
        assert tx["outs"][0]["script"].startswith("a914")
        assert tx["outs"][0]["script"].endswith("87")
        payload_hex = _opreturn_payload_via_reader(tx["outs"][1]["script"])
        assert payload_hex == chunk.hex()
        reassembled += bytes.fromhex(payload_hex)
    assert reassembled == body


def test_cadena_make_tx_embeds_first_chunk():
    """Step-mode single-key Cadena: make_tx stages the next chunk's OP_RETURN."""
    body = _real_body()
    c = inscriptions.Cadena(PRIV, body, dict(UTXO), TIP)
    c.make_tx()
    tx = pydoge.deserialize(pydoge.serialize(c.signed_inscribed_tx))
    assert len(tx["outs"]) == 2
    assert _opreturn_payload_via_reader(tx["outs"][1]["script"]) == body[:80].hex()


# --- node-side validation (opt-in; no funds spent) --------------------------

def _node_or_skip():
    if not os.getenv("QUIPU_RPC_TESTS"):
        pytest.skip("set QUIPU_RPC_TESTS=1 to run node-backed checks")
    try:
        node.current_block_height()
    except Exception as exc:
        pytest.skip(f"node unreachable: {exc}")


def test_node_decodes_staged_opreturn():
    """Dogecoin Core itself decodes the staged tx as a standard nulldata
    OP_RETURN carrying the exact payload. Read-only: decoderawtransaction does
    not touch the wallet or chain, and the fake funding utxo is never spent."""
    _node_or_skip()
    body = _real_body()
    atom = inscriptions.CadenaAtom(PRIV, body, dict(UTXO), TIP)
    atom.precompute()

    decoded = node.rpc_request("decoderawtransaction", [atom.txns[0]])
    assert decoded["txid"] == atom.txn_ids[0]          # node agrees on the txid
    op_out = decoded["vout"][1]
    spk = op_out["scriptPubKey"]
    assert spk["type"] == "nulldata"                   # node sees a standard OP_RETURN
    assert op_out["value"] == 0
    assert spk["asm"].startswith("OP_RETURN")
    # node-decoded payload hex equals the chunk we embedded
    assert _opreturn_payload_via_reader(spk["hex"]) == body[:80].hex()


def test_node_testmempoolaccept_structure():
    """testmempoolaccept must reject for missing-inputs (the funding utxo is
    fake) — NOT for a malformed/oversized tx. That rejection reason is itself
    proof the transaction structure + OP_RETURN are valid and standard."""
    _node_or_skip()
    atom = inscriptions.CadenaAtom(PRIV, _real_body(), dict(UTXO), TIP)
    atom.precompute()
    try:
        res = node.rpc_request("testmempoolaccept", [[atom.txns[0]]])[0]
    except RuntimeError as exc:
        if "Method not found" in str(exc):
            pytest.skip("node lacks testmempoolaccept (pre-1.21 Dogecoin Core)")
        raise
    assert res["allowed"] is False
    reason = res.get("reject-reason", "")
    assert "missing" in reason or "inputs" in reason, f"unexpected reject: {reason!r}"
