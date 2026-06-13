"""The foundation swap, proven: pydoge reproduces `cryptos` byte-for-byte for
every operation the quipu toolkit uses. This is the migration oracle — once it
is green, the cryptos dependency can leave the toolkit entirely.

`cryptos` is a test-only dependency (see pyproject [test]); skip if absent.
"""

import pytest

cryptos = pytest.importorskip("cryptos")

import pydoge
from pydoge import keys, transaction as tx, script

doge = cryptos.Doge()

# a fixed 64-hex test private key (NOT a real key). 64-hex → uncompressed in
# both libraries, matching the apocrypha-era on-chain convention.
PRIV = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

# a real on-chain Dogecoin tx (the heal diamond's final join) for serialize parity
JOIN_HEX = (
    "010000000219815194179d49d383a64d75942d427343bd6a2489f22f111b1d24bc0b4b6e6c"
    "000000008b483045022100eb973972a519d16e359ca3474d50ba3813299a535056ea39e398"
    "acafb592d35102203f61feda0a29f847685339e3528b69ff78fa50d9ff48199841e7fec277"
    "22baff0141047c88e9a4df6e9f45656c10bf66f28e28be235a15b64820b254f1b9eb273831"
    "4e6769f5c94da0c7640ffe76dcffca053b07a0804cd53a1c51ad03bfe0133ce8c5ffffffff"
    "87539a5918c05ac6dcf6b68120cd57a38e558cf22db274fad5d31c70187cb9b4000000008a"
    "473044022038c5fbdaf9aeada04f92ab93b48a6257ea362e11cb5604f5b2e326555891086a"
    "02205626838e90cdd19cbdf7ba58bb54705f53801da112dd1a2aac2cd3d04ba819ca014104"
    "7c88e9a4df6e9f45656c10bf66f28e28be235a15b64820b254f1b9eb2738314e6769f5c94da"
    "0c7640ffe76dcffca053b07a0804cd53a1c51ad03bfe0133ce8c5ffffffff012adf3832060"
    "000001976a914144739367df0ff8d1c61d03704298d49cf93ef3f88ac00000000"
)


def test_pubtoaddr_matches_cryptos():
    for compressed in (False, True):
        pub = keys.privtopub(PRIV, compressed=compressed)
        assert keys.pubtoaddr(pub) == doge.pubtoaddr(pub)


def test_privtoaddr_uncompressed_matches_cryptos():
    assert keys.privtoaddr(PRIV, compressed=False) == doge.privtoaddr(PRIV)


def test_serialize_roundtrip_matches_cryptos():
    assert pydoge.serialize(pydoge.deserialize(JOIN_HEX)) == JOIN_HEX
    assert cryptos.serialize(cryptos.deserialize(JOIN_HEX)) == JOIN_HEX


def test_full_build_sign_is_byte_identical():
    """The whole pipeline — build, sign, serialize — must match cryptos exactly."""
    ins = [{"output": "ab" * 32 + ":0", "value": 100000}]
    addr = keys.privtoaddr(PRIV, compressed=False)
    outs = [{"value": 90000, "address": addr}]
    pd = tx.serialize(tx.signall(tx.mktx(ins, outs), PRIV))
    cr = cryptos.serialize(doge.signall(doge.mktx(ins, outs), PRIV))
    assert pd == cr


def test_multisig_address_matches_cryptos():
    pubs = [keys.privtopub(PRIV, compressed=False),
            keys.privtopub("11" * 32, compressed=False)]
    ref = doge.mk_multisig_address(pubs, 2)
    if isinstance(ref, (tuple, list)):
        ref = ref[-1]
    assert script.mk_multisig_address(pubs, 2) == ref
