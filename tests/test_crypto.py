"""Hermetic tests for the encrypted-quipu layer (crypto.py).

The ECIES bodies use random nonces/session keys, so raw ciphertext isn't
reproducible — instead we assert (a) deterministic structural headers are
byte-identical to the monolith, and (b) **cross-decrypt** wire-compatibility:
something built by the new code reads back under the old code and vice-versa.
The deterministic primitives (keydrop, pubkey parsing, KDF) are compared
byte-for-byte. Node-backed resolvers (get_address_pubkeys, array_dec_from_txn)
need a live wallet and are exercised separately. Skips if the monolith or
cryptos isn't importable.
"""

import hashlib
import os
import sys

import ecies
import pytest

import pydoge
from colegio import crypto

_HERE = os.path.dirname(os.path.abspath(__file__))
_MONO_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "Colegio_Invisible"))
if _MONO_DIR not in sys.path:
    sys.path.insert(0, _MONO_DIR)
try:
    import colegio_tools as old
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"monolith colegio_tools not importable: {exc}",
                allow_module_level=True)

KEY32 = bytes(range(32))
INNER_HEADER = b"\xc1\xdd\x00\x01\x03\x00\x01\x00\x10\x00\x10\x02|title|"
INNER_BODY = b"the plaintext inner body bytes \x00\x01\x02"


# --- deterministic primitives (byte-identical) ------------------------------

def test_strip_pub_prefix_matches_monolith():
    p_un = "04" + "ab" * 64  # 130-hex Bitcoin uncompressed
    assert crypto._strip_pub_prefix(p_un) == old._strip_pub_prefix(p_un) == "ab" * 64
    p_eth = "cd" * 64  # 128-hex eth_keys form
    assert crypto._strip_pub_prefix(p_eth) == old._strip_pub_prefix(p_eth) == "cd" * 64


def test_coerce_aes_key_matches_monolith():
    assert crypto._coerce_aes_key(KEY32) == old._coerce_aes_key(KEY32) == KEY32
    assert crypto._coerce_aes_key("pw") == old._coerce_aes_key("pw") \
        == hashlib.sha256(b"pw").digest()


def test_parse_multisig_redeem_uncompressed_matches_monolith():
    p1 = pydoge.privtopub("11" * 32, compressed=False)
    p2 = pydoge.privtopub("22" * 32, compressed=False)
    redeem = bytes.fromhex(pydoge.mk_multisig_script([p1, p2], 2))
    assert crypto._parse_multisig_redeem(redeem) \
        == old._parse_multisig_redeem(redeem) == [p1, p2]


def test_parse_multisig_redeem_compressed_matches_monolith():
    p1 = pydoge.privtopub("11" * 32, compressed=True)
    p2 = pydoge.privtopub("22" * 32, compressed=True)
    redeem = bytes.fromhex(pydoge.mk_multisig_script([p1, p2], 2))
    new_pubs = crypto._parse_multisig_redeem(redeem)
    assert new_pubs == old._parse_multisig_redeem(redeem)
    assert all(len(p) == 130 for p in new_pubs)  # uncompressed on the way out


def test_keydrop_build_parse_matches_monolith():
    txid = "ab" * 32
    h_new, b_new = crypto.build_keydrop_quipu(txid, KEY32, b"|kd|")
    h_old, b_old = old.build_keydrop_quipu(txid, KEY32, b"|kd|")
    assert (h_new, b_new) == (h_old, b_old)
    assert crypto.parse_keydrop_quipu(h_new, b_new) == (txid, KEY32)
    assert crypto.parse_keydrop_quipu(h_new, b_new) == old.parse_keydrop_quipu(h_new, b_new)


# --- ECDH/HKDF (deterministic given keys) -----------------------------------

def test_shared_key_matches_monolith():
    a = ecies.utils.generate_eth_key()
    b = ecies.utils.generate_eth_key()
    assert crypto.shared_key(a, b.public_key) == old.shared_key(a, b.public_key)


# --- encrypted families: header determinism + cross-decrypt -----------------

def test_aes_sealed_header_deterministic_and_cross_decrypt():
    key = "passphrase"
    oh_new, _ = crypto.build_aes_sealed_quipu(INNER_HEADER, INNER_BODY, key)
    oh_old, _ = old.build_aes_sealed_quipu(INNER_HEADER, INNER_BODY, key)
    assert oh_new == oh_old  # structural header is deterministic

    # new build → old read, and old build → new read (wire-compatible)
    oh, ob = crypto.build_aes_sealed_quipu(INNER_HEADER, INNER_BODY, key)
    assert old.read_aes_sealed_quipu(oh, ob, key) == (INNER_HEADER, INNER_BODY)
    oh2, ob2 = old.build_aes_sealed_quipu(INNER_HEADER, INNER_BODY, key)
    assert crypto.read_aes_sealed_quipu(oh2, ob2, key) == (INNER_HEADER, INNER_BODY)


def test_broadcast_header_deterministic_and_roundtrip():
    author = ecies.utils.generate_eth_key()
    r1 = ecies.utils.generate_eth_key()
    r2 = ecies.utils.generate_eth_key()
    inner_struct = b"\x03\x00\x01\x00\x10\x00\x10\x02"  # type,tone,color,LL,WW,B
    title = b"|img|"
    body = b"image bitstream bytes"
    recips = [r1.public_key, r2.public_key]

    oh_new, _ = crypto.build_broadcast_quipu(inner_struct, title, body, author, recips)
    oh_old, _ = old.build_broadcast_quipu(inner_struct, title, body, author, recips)
    assert oh_new == oh_old  # deterministic header (tone byte dropped, N + title)

    oh, ob = crypto.build_broadcast_quipu(inner_struct, title, body, author, recips)
    for r in (r1, r2):
        ih, ib = crypto.read_broadcast_quipu(oh, ob, r, author.public_key)
        assert ib == body
        # the same wire bytes decode under the monolith too
        ih_old, ib_old = old.read_broadcast_quipu(oh, ob, r, author.public_key)
        assert (ih_old, ib_old) == (ih, ib)


def test_apply_keydrop_aes_sealed_matches_monolith():
    oh, ob = crypto.build_aes_sealed_quipu(INNER_HEADER, INNER_BODY, KEY32)
    assert crypto.apply_keydrop(oh, ob, KEY32) == (INNER_HEADER, INNER_BODY)
    assert crypto.apply_keydrop(oh, ob, KEY32) == old.apply_keydrop(oh, ob, KEY32)
