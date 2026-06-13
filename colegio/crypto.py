"""crypto — ECIES box crypto, AES seal, and the key-drop primitive.

The encrypted-quipu layer. Two encrypted families live under header byte 4 = 0x0e:

    0x0e 0x03 ...    Broadcast: N per-recipient ECDH-locked session-key copies,
                     then an AES-encrypted body (in use on chain).
    0x0e 0xae ...    AES-sealed: no envelopes. Body is AES-encrypted with a key
                     supplied out-of-band (passphrase via SHA-256, or raw 32-byte
                     key). Wraps any plaintext inner type.

The 0x0e 0x0e 0x0d "key drop" releases the AES key for a previously-broadcast
0e 03 quipu, or for a 0e ae quipu — same primitive.

ECDH/HKDF via coincurve; symmetric crypto via eciespy (`ecies.sym_*`) and
pycryptodome's HKDF. Keys are eth_keys PrivateKey/PublicKey objects. No cryptos.
"""

import hashlib

import coincurve
import ecies
import eth_keys
from coincurve.utils import get_valid_secret
from Crypto.Hash import SHA256
from Crypto.Protocol.KDF import HKDF

from . import node
from .imaging import read_image_data
from .reading import read_quipu

AES_KEY_BYTES_LEN = 32


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


def build_aes_sealed_quipu(inner_header_bytes, inner_body_bytes, password_or_key):
    """Wrap a plaintext quipu into the 0x0e 0xae AES-sealed form.

    The wrap is structural: insert `0e ae` between the c1dd0001 magic+version
    and the inner type byte. Title (the |…| field) and all inner-type
    structural fields stay in their relative positions, just shifted by
    two bytes. The unwrap is symmetric.

    Returns (outer_header_bytes, outer_body_bytes).
    """
    if inner_header_bytes[:4] != b"\xc1\xdd\x00\x01":
        raise ValueError("inner header must start with c1dd 0001")
    outer_header = b"\xc1\xdd\x00\x01\x0e\xae" + inner_header_bytes[4:]
    outer_body = aes_encrypt_bytes(inner_body_bytes, password_or_key)
    return outer_header, outer_body


def read_aes_sealed_quipu(outer_header_bytes, outer_body_bytes, password_or_key):
    """Unwrap a 0x0e 0xae quipu. Returns (inner_header_bytes, inner_body_bytes)
    in plaintext-quipu shape so existing per-type readers handle the result."""
    if outer_header_bytes[:6] != b"\xc1\xdd\x00\x01\x0e\xae":
        raise ValueError("not an AES-sealed quipu (expected c1dd 0001 0e ae prefix)")
    inner_header = b"\xc1\xdd\x00\x01" + outer_header_bytes[6:]
    inner_body = aes_decrypt_bytes(outer_body_bytes, password_or_key)
    return inner_header, inner_body


def build_broadcast_quipu(inner_header_struct, title_field, inner_body_bytes,
                          author_privkey, recipient_pubkeys):
    """Build a 0x0e 0x03 broadcast-encrypted image quipu (nb17 format).

    Inputs:
      inner_header_struct : bytes 4+ of a plaintext image header — i.e.
        [type][tone][color][LL][WW][B]. Tone is dropped in the broadcast
        layout (byte 5 in broadcast carries the inner type instead).
      title_field         : e.g. b'|My Image|', with bordering pipes.
      inner_body_bytes    : raw inner content (the image bitstream).
      author_privkey      : eth_keys PrivateKey of the inscriber.
      recipient_pubkeys   : list of eth_keys PublicKey objects.

    Layout (image case):
      header: c1dd 0001 0e <inner_type=03> <color> LL WW B Nrecip <title>
      body  : [Nrecip × 64-byte session-key copies][AES(session, inner_body)]
    """
    inner_type = inner_header_struct[0:1]
    structural_tail = inner_header_struct[2:]  # drop tone byte
    N = len(recipient_pubkeys)
    if N > 255:
        raise ValueError("N_recip must fit in a single byte")
    outer_header = (b"\xc1\xdd\x00\x01\x0e" + inner_type + structural_tail
                    + bytes([N]) + title_field)
    session = get_valid_secret()
    envelopes = b"".join(
        ecies.sym_encrypt(shared_key(author_privkey, pub), session)
        for pub in recipient_pubkeys
    )
    body_ciphertext = ecies.sym_encrypt(session, inner_body_bytes)
    return outer_header, envelopes + body_ciphertext


def read_broadcast_quipu(outer_header_bytes, outer_body_bytes, my_privkey, author_pubkey):
    """Decrypt a 0x0e 0x03 broadcast quipu by trying each envelope against
    `my_privkey`. Returns (inner_header_bytes, inner_body_bytes) where the
    inner header is synthesized as plaintext-image-shaped (with a placeholder
    tone byte of 0x00, since broadcast drops the tone byte at write time)."""
    if outer_header_bytes[4:5] != b"\x0e":
        raise ValueError("not an encrypted quipu (byte 4 != 0x0e)")
    inner_type = outer_header_bytes[5:6]
    color_lwwb = outer_header_bytes[6:12]
    N = outer_header_bytes[12]
    title = outer_header_bytes[13:]
    inner_header = (b"\xc1\xdd\x00\x01" + inner_type + b"\x00"
                    + color_lwwb + title)
    envelopes = [outer_body_bytes[i * 64:(i + 1) * 64] for i in range(N)]
    cipher_body = outer_body_bytes[N * 64:]
    sk = shared_key(my_privkey, author_pubkey)
    last_err = None
    for env in envelopes:
        try:
            session = ecies.sym_decrypt(sk, env)
            plain = ecies.sym_decrypt(session, cipher_body)
            return inner_header, plain
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("no envelope decrypted with this key")


def build_keydrop_quipu(target_txid_hex, aes_key, title_field=b""):
    """Build a 0x0e 0x0e 0x0d key-drop quipu releasing `aes_key` for the
    encrypted quipu at `target_txid_hex`. Body layout per nb18:
    [32-byte target txid bytes][32-byte AES key]. The txid is stored as
    bytes.fromhex(displayed_txid) — display-endian, not Bitcoin-internal."""
    if len(aes_key) != 32:
        raise ValueError("aes_key must be 32 bytes")
    header = b"\xc1\xdd\x00\x01\x0e\x0e\x0d" + title_field
    body = bytes.fromhex(target_txid_hex) + aes_key
    return header, body


def parse_keydrop_quipu(header_bytes, body_bytes):
    """Inverse of build_keydrop_quipu. Returns (target_txid_hex, aes_key)."""
    if header_bytes[4:7] != b"\x0e\x0e\x0d":
        raise ValueError("not a key-drop quipu (header prefix c1dd0001 0e 0e 0d expected)")
    if len(body_bytes) < 64:
        raise ValueError("key-drop body too short (need 64 bytes)")
    return body_bytes[:32].hex(), body_bytes[32:64]


def find_keydrop_for(encrypted_txid_hex, quipus, df_outputs):
    """Scan a list of quipu rows for a key-drop whose body's first 32 bytes
    match the given encrypted-quipu txid (display-endian per nb18).

    `quipus` is an iterable of dict-likes with a 'root_txid' field.
    Returns (keydrop_row, aes_key_bytes) or None.
    """
    target = bytes.fromhex(encrypted_txid_hex)
    for q in quipus:
        try:
            head_hex, body_hex = read_quipu(q["root_txid"], df_outputs)
        except Exception:
            continue
        head = bytes.fromhex(head_hex)
        if len(head) < 7 or head[4:7] != b"\x0e\x0e\x0d":
            continue
        body = bytes.fromhex(body_hex)
        if len(body) < 64:
            continue
        if body[:32] == target:
            return q, body[32:64]
    return None


def apply_keydrop(target_header_bytes, target_body_bytes, aes_key):
    """Apply a released AES key to an encrypted quipu's header+body. Handles
    both 0e 03 (broadcast — skip N_recip envelopes) and 0e ae (AES-sealed —
    decrypt directly) targets. Returns plaintext (inner_header, inner_body)."""
    if target_header_bytes[4:5] != b"\x0e":
        raise ValueError("target is not encrypted (byte 4 != 0x0e)")
    sub = target_header_bytes[5:6]
    if sub == b"\xae":
        return read_aes_sealed_quipu(target_header_bytes, target_body_bytes, aes_key)
    if sub == b"\x03":
        N = target_header_bytes[12]
        cipher_body = target_body_bytes[N * 64:]
        plain = ecies.sym_decrypt(aes_key, cipher_body)
        inner_type = target_header_bytes[5:6]
        color_lwwb = target_header_bytes[6:12]
        title = target_header_bytes[13:]
        inner_header = (b"\xc1\xdd\x00\x01" + inner_type + b"\x00"
                        + color_lwwb + title)
        return inner_header, plain
    raise ValueError(f"unsupported encrypted sub-family byte: {sub.hex()}")
