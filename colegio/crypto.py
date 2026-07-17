"""crypto — ECIES box crypto, AES seal, and the key-drop primitive.

The encrypted-quipu layer (family 0x0e). Since the July 2026 wire
reconciliation (`Colegio_Invisible/docs/design/encrypted-wire-reconciliation.md`)
there is ONE implementation of the 0x0e wire format —
`Colegio_Invisible/canonical/encrypted.py` — and this module, like the
monolith's `colegio_tools`, is a thin surface over it:

    build_*  emit the CANONICAL v1 layout (0e <tone> ae/ec/0d …). The
             pre-canonical nb17/nb18 layouts are never written again.
    read_*   accept BOTH eras — the four pre-canonical inscriptions
             (d0209a, d68175 broadcasts; 89b51b, f278e4 keydrops) stay
             legible forever via the canonical module's legacy readers.

The canonical module is resolved through the same seam the invisible
client uses: `$COLEGIO_INVISIBLE` (defaulting to the sibling checkout).
The node-backed helpers (pubkey recovery, address scanning) and the raw
AES byte primitives are self-contained and need no seam.

ECDH/HKDF via coincurve; symmetric crypto via eciespy (`ecies.sym_*`) and
pycryptodome's HKDF. Keys are eth_keys PrivateKey/PublicKey objects
(coincurve objects also accepted). No cryptos.
"""

import hashlib
import os
import sys

import coincurve
import ecies
import eth_keys
from Crypto.Hash import SHA256
from Crypto.Protocol.KDF import HKDF

from . import node
from .imaging import read_image_data
from .reading import read_quipu

AES_KEY_BYTES_LEN = 32


def _enc():
    """The canonical encrypted module — the ONE 0x0e wire implementation
    (lazy import via the `$COLEGIO_INVISIBLE` seam)."""
    cinv = os.environ.get("COLEGIO_INVISIBLE") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Colegio_Invisible")
    canon = os.path.join(cinv, "canonical")
    if not os.path.isdir(canon):
        raise RuntimeError(
            f"canonical encrypted module not found (looked in {canon}); "
            f"set COLEGIO_INVISIBLE to the Colegio_Invisible checkout")
    if canon not in sys.path:
        sys.path.insert(0, canon)
    import encrypted as E
    return E


def _cc_priv(priv):
    """eth_keys PrivateKey / coincurve.PrivateKey / 32 bytes → coincurve."""
    if isinstance(priv, coincurve.PrivateKey):
        return priv
    if hasattr(priv, "to_bytes"):
        return coincurve.PrivateKey(priv.to_bytes())
    return coincurve.PrivateKey(bytes(priv))


def _cc_pub(pub):
    """eth_keys PublicKey / coincurve.PublicKey / serialized bytes → coincurve."""
    if isinstance(pub, coincurve.PublicKey):
        return pub
    if hasattr(pub, "to_compressed_bytes"):
        return coincurve.PublicKey(pub.to_compressed_bytes())
    return coincurve.PublicKey(bytes(pub))


# --- ECIES helpers ----------------------------------------------------------
def shared_key(prvKey, pubKey):
    cc_prv = coincurve.PrivateKey(prvKey.to_bytes())
    cc_pub = coincurve.PublicKey(pubKey.to_compressed_bytes())
    return HKDF(cc_pub.multiply(cc_prv.secret).format(), AES_KEY_BYTES_LEN, b"", SHA256)


def get_txn_pub_from_node(txn_ident):
    """Recover the pubkey used to sign the first input of a tx via the node.

    Uses gettransaction + decoderawtransaction so it works in pruned mode for
    wallet-relevant txs (getrawtransaction would need -txindex).
    """
    wallet_tx = node.rpc_request("gettransaction", [txn_ident, True])
    raw = node.rpc_request("decoderawtransaction", [wallet_tx["hex"]])
    asm = raw["vin"][0]["scriptSig"]["asm"]
    # asm format: "<sig> <pubkey>" — pubkey is the last token, uncompressed = 130 hex chars
    return asm.split()[-1]


def _strip_pub_prefix(pub_hex):
    """Accept 128-hex (eth_keys form) or 130-hex with leading '04' (Bitcoin
    uncompressed form). Returns the 128-hex eth_keys form."""
    pub_hex = pub_hex.strip().lower()
    if len(pub_hex) == 130 and pub_hex.startswith("04"):
        return pub_hex[2:]
    if len(pub_hex) == 128:
        return pub_hex
    raise ValueError(f"unexpected pubkey hex length: {len(pub_hex)} (need 128 or 130)")


def get_address_pubkeys(address, max_scan=2000):
    """Resolve a Dogecoin address to its underlying secp256k1 pubkey(s).

    P2PKH (single-key) addresses → returns a list of length 1.
    P2SH multisig addresses → returns the list of component pubkeys parsed
    from the redeem script.

    Strategy: scan the wallet's `listtransactions ['*' ...]` history (which
    includes any watched address with importaddress) for an input whose
    decoded scriptSig comes from this address, then either:
      - extract the trailing pubkey from a P2PKH scriptSig ('<sig> <pubkey>')
      - extract the redeem script from a P2SH scriptSig (last asm token)
        and parse its '<m> <pk1> <pk2> ... <n> OP_CHECKMULTISIG' form

    Returns a list of pubkey hex strings (128 chars, eth_keys form).

    Raises RuntimeError if no spending tx is found within max_scan recent
    wallet txs — typical when the recipient address has never spent on
    chain, or isn't watched.
    """
    txs = node.rpc_request("listtransactions", ["*", max_scan, 0, True])
    seen_txids = set()
    for t in reversed(txs):
        txid = t.get("txid")
        if not txid or txid in seen_txids:
            continue
        seen_txids.add(txid)
        try:
            wtx = node.rpc_request("gettransaction", [txid, True])
            raw = node.rpc_request("decoderawtransaction", [wtx["hex"]])
        except Exception:
            continue
        for vin in raw.get("vin", []):
            prev_txid = vin.get("txid")
            prev_vout = vin.get("vout")
            if prev_txid is None or prev_vout is None:
                continue
            # Identify the address that signed this input
            try:
                prev_wtx = node.rpc_request("gettransaction", [prev_txid, True])
                prev_raw = node.rpc_request("decoderawtransaction", [prev_wtx["hex"]])
            except Exception:
                continue
            try:
                spk = prev_raw["vout"][prev_vout]["scriptPubKey"]
            except (IndexError, KeyError):
                continue
            addrs = spk.get("addresses") or []
            if address not in addrs:
                continue

            asm = vin.get("scriptSig", {}).get("asm", "")
            tokens = asm.split()
            if not tokens:
                continue

            spk_type = spk.get("type")
            if spk_type == "pubkeyhash":
                # P2PKH: scriptSig is '<sig> <pubkey>'
                pub_hex = tokens[-1]
                try:
                    return [_strip_pub_prefix(pub_hex)]
                except ValueError:
                    continue
            if spk_type == "scripthash":
                # P2SH: scriptSig is 'OP_0 <sig1> ... <redeem_script>'
                # The redeem script is the last asm token; parse it as a
                # standard multisig 'm <pk1>..<pkN> n OP_CHECKMULTISIG'.
                # The 'asm' for the redeem script in OP_PUSHDATA form is
                # the last hex blob — need to decode it as script tokens.
                redeem_hex = tokens[-1]
                # Manual parse: redeem script is the same hex; iterate ops.
                try:
                    redeem = bytes.fromhex(redeem_hex)
                except ValueError:
                    continue
                pubs = _parse_multisig_redeem(redeem)
                if pubs:
                    return [_strip_pub_prefix(p) for p in pubs]
                continue
            # Other script types (witness, etc.) not handled
    raise RuntimeError(
        f"could not resolve pubkey(s) for address {address}: no spending "
        f"tx found in the last {max_scan} wallet transactions. The address "
        f"must have spent at least once on chain, and either be in the "
        f"wallet or have been involved in a tx the wallet has seen."
    )


def _parse_multisig_redeem(script_bytes):
    """Parse a standard multisig redeem script and return the list of pubkey
    hex strings. Returns [] if the script isn't a valid m-of-n multisig."""
    if len(script_bytes) < 4:
        return []
    # Opcodes: OP_1..OP_16 are 0x51..0x60; OP_CHECKMULTISIG is 0xae.
    OP_CHECKMULTISIG = 0xae
    if script_bytes[-1] != OP_CHECKMULTISIG:
        return []
    m_byte = script_bytes[0]
    n_byte = script_bytes[-2]
    if not (0x51 <= m_byte <= 0x60 and 0x51 <= n_byte <= 0x60):
        return []
    n = n_byte - 0x50
    # Walk pushes between [1] and [-2]
    pos = 1
    end = len(script_bytes) - 2
    pubs = []
    while pos < end:
        push_len = script_bytes[pos]
        # Standard secp256k1 pubkeys: 0x21 (compressed 33B) or 0x41 (uncompressed 65B)
        if push_len in (0x21, 0x41):
            pos += 1
            if pos + push_len > end:
                return []
            pubs.append(script_bytes[pos:pos + push_len].hex())
            pos += push_len
        else:
            return []
    if len(pubs) != n:
        return []
    # Normalize: if any pubkey is compressed, uncompress it via coincurve so
    # we always return uncompressed-eth_keys form. (eth_keys' PublicKey
    # constructor wants 64 bytes, no prefix.)
    out = []
    for p in pubs:
        if len(p) == 66:  # compressed
            cc = coincurve.PublicKey(bytes.fromhex(p))
            uncompressed = cc.format(compressed=False).hex()
            out.append(uncompressed)
        else:
            out.append(p)
    return out


def array_dec_from_txn(txn_ident, prvKey_input, index_key, df_outputs):
    """Decrypt an image-quipu addressed to one of N recipients."""
    hex_header, body_hex = read_quipu(txn_ident, df_outputs)
    enc_bytes = bytes.fromhex(body_hex)
    N_keys = int(hex_header[24:26])
    zip_keys = [enc_bytes[i * 64:(i + 1) * 64] for i in range(N_keys)]
    zip_data = enc_bytes[N_keys * 64:]
    pub_hex = get_txn_pub_from_node(txn_ident)
    txn_pub = eth_keys.keys.PublicKey(bytes.fromhex(pub_hex))
    sk = shared_key(prvKey_input, txn_pub)
    session = ecies.sym_decrypt(sk, zip_keys[index_key])
    data = ecies.sym_decrypt(session, zip_data)
    return hex_header, read_image_data(hex_header, data)


# --- Encrypted-quipu wrappers -----------------------------------------------
def _coerce_aes_key(password_or_key):
    """32-byte AES key passthrough, else SHA-256(passphrase) — same KDF
    convention as scripts/aes_encrypt.py and nb02's aes_encrypt_file."""
    if isinstance(password_or_key, (bytes, bytearray)) and len(password_or_key) == 32:
        return bytes(password_or_key)
    if isinstance(password_or_key, str):
        return hashlib.sha256(password_or_key.encode()).digest()
    raise TypeError("password_or_key must be a 32-byte key or a passphrase string")


def aes_encrypt_bytes(plain_bytes, password_or_key):
    """Byte-level analog of scripts/aes_encrypt.py:aes_encrypt_file."""
    return ecies.sym_encrypt(key=_coerce_aes_key(password_or_key), plain_text=plain_bytes)


def aes_decrypt_bytes(cipher_bytes, password_or_key):
    return ecies.sym_decrypt(key=_coerce_aes_key(password_or_key), cipher_text=cipher_bytes)


def build_aes_sealed_quipu(inner_header_bytes, inner_body_bytes, password_or_key,
                           *, title="", tone=0x00):
    """AES-seal a plaintext quipu. Emits the CANONICAL layout
    (0e <tone> ae <variant>, body = AES(key, length-framed inner header+body));
    the whole inner quipu, header included, is sealed — unlike the old
    splice, which left the inner header in cleartext.

    `password_or_key`: 32-byte raw key, or passphrase string (SHA-256 KDF).
    Returns (outer_header_bytes, outer_body_bytes).
    """
    E = _enc()
    key = password_or_key if isinstance(password_or_key, str) else bytes(password_or_key)
    return E.build_aes_quipu(inner_header_bytes, inner_body_bytes, key,
                             title=title, tone=tone)


def read_aes_sealed_quipu(outer_header_bytes, outer_body_bytes, password_or_key):
    """Unwrap an AES-sealed quipu of EITHER era (canonical 0e <tone> ae, or
    the pre-canonical 0e ae splice). Returns (inner_header, inner_body) in
    plaintext-quipu shape so existing per-type readers handle the result."""
    E = _enc()
    if E.classify_encrypted(outer_header_bytes) == "legacy_aes":
        return E.read_legacy_aes_sealed(outer_header_bytes, outer_body_bytes,
                                        password_or_key)
    parsed = E.read_encrypted_quipu(outer_header_bytes, outer_body_bytes,
                                    key=password_or_key)
    if "inner_header" not in parsed:
        raise ValueError(f"not an AES-openable quipu (sub {parsed.get('sub_name')})")
    return parsed["inner_header"], parsed["inner_body"]


def build_broadcast_quipu(inner_header_struct, title_field, inner_body_bytes,
                          author_privkey, recipient_pubkeys):
    """Broadcast-seal a quipu to N recipients. Emits the CANONICAL ECIES
    layout (0e <tone> ec 00, body = <N:1> + N×64B envelopes + AES(session,
    framed inner)) — the pre-canonical nb17 layout is never written again.

    Historical signature, kept for callers:
      inner_header_struct : bytes 4+ of the plaintext inner header, up to
                            (not including) the |title| field
      title_field         : e.g. b'|My Image|', with bordering pipes; also
                            reused as the outer public title
      inner_body_bytes    : raw inner content
      author_privkey      : eth_keys or coincurve PrivateKey
      recipient_pubkeys   : list of eth_keys or coincurve PublicKey
    """
    E = _enc()
    inner_header = (b"\xc1\xdd\x00\x01" + bytes(inner_header_struct)
                    + bytes(title_field))
    outer_title = bytes(title_field).strip(b"|").decode("utf-8", "replace")
    return E.build_ecies_quipu(inner_header, inner_body_bytes,
                               _cc_priv(author_privkey),
                               [_cc_pub(p) for p in recipient_pubkeys],
                               title=outer_title)


def read_broadcast_quipu(outer_header_bytes, outer_body_bytes, my_privkey, author_pubkey):
    """Decrypt a broadcast quipu of EITHER era (canonical 0e <tone> ec, or
    pre-canonical 0e 03) by trying each envelope against `my_privkey`.
    Returns (inner_header_bytes, inner_body_bytes); for the legacy layout
    the inner header is synthesized plaintext-image-shaped (placeholder
    tone 0x00 — the old writer dropped the tone byte)."""
    E = _enc()
    parsed = E.read_encrypted_quipu(outer_header_bytes, outer_body_bytes,
                                    my_privkey=_cc_priv(my_privkey),
                                    author_pubkey=_cc_pub(author_pubkey))
    if "inner_header" not in parsed:
        raise ValueError(f"could not decrypt (sub {parsed.get('sub_name')})")
    return parsed["inner_header"], parsed["inner_body"]


def build_keydrop_quipu(target_txid_hex, aes_key, title_field=b""):
    """Release `aes_key` for the encrypted quipu at `target_txid_hex`.
    Emits the CANONICAL keydrop (0e <tone> 0d 00, u16-count body) with a
    single anonymous drop — the pre-canonical 0e 0e 0d layout is never
    written again. The txid is display-endian, as before."""
    E = _enc()
    title = bytes(title_field).strip(b"|").decode("utf-8", "replace")
    return E.build_keydrop_quipu([("", target_txid_hex, bytes(aes_key))],
                                 title=title)


def parse_keydrop_quipu(header_bytes, body_bytes):
    """First released (target_txid_hex, aes_key) of a keydrop of EITHER era.
    (Canonical keydrops may carry several drops — use the canonical module's
    read_encrypted_quipu for the full list.)"""
    E = _enc()
    parsed = E.read_encrypted_quipu(header_bytes, body_bytes)
    drops = parsed.get("drops")
    if not drops:
        raise ValueError(f"not a key-drop quipu (sub {parsed.get('sub_name')})")
    return drops[0]["ref_txid"], drops[0]["key"]


def find_keydrop_for(encrypted_txid_hex, quipus, df_outputs):
    """Scan a list of quipu rows for a keydrop (either era) releasing a key
    for the given encrypted-quipu txid.

    `quipus` is an iterable of dict-likes with a 'root_txid' field.
    Returns (keydrop_row, aes_key_bytes) or None.
    """
    E = _enc()
    for q in quipus:
        try:
            head_hex, body_hex = read_quipu(q["root_txid"], df_outputs)
            head = bytes.fromhex(head_hex)
            if len(head) < 7 or head[4:5] != b"\x0e":
                continue
            parsed = E.read_encrypted_quipu(head, bytes.fromhex(body_hex))
        except Exception:
            continue
        for d in parsed.get("drops") or []:
            if d["ref_txid"] == encrypted_txid_hex:
                return q, d["key"]
    return None


def apply_keydrop(target_header_bytes, target_body_bytes, aes_key):
    """Apply a released 32-byte key to a sealed quipu of EITHER era
    (canonical ae/ec, legacy broadcast, legacy AES splice). Returns
    plaintext (inner_header, inner_body)."""
    return _enc().open_with_key(target_header_bytes, target_body_bytes, aes_key)
