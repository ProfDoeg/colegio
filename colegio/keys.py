"""keys — cinv keyfile / pubkey / address I/O, and key generation.

Encrypted keyfiles (`*_prv.enc`) are AES-sealed with SHA-256(password) as the
key (the project's deliberate minimal-crypto convention). Decryption yields an
eth_keys PrivateKey. Address derivation uses pydoge (not cryptos).
"""

import getpass
import hashlib
import os

import ecies
import eth_keys
import qrcode
import pydoge


def save_privkey(privkey, privkey_filepath, password=None):
    if password is None:
        while True:
            password = getpass.getpass("Input password for encrypting keyfile: ")
            if password == getpass.getpass("Repeat password: "):
                print("\nPasswords match...")
                break
            print("\nPasswords do not match...")
    encrypted = ecies.sym_encrypt(
        key=hashlib.sha256(password.encode()).digest(),
        plain_text=privkey.to_bytes(),
    )
    with open(privkey_filepath, "wb") as f:
        f.write(encrypted)
    print(f"Password protected file written to {privkey_filepath}")


def import_privKey(privkey_filepath, password=None):
    if password is None:
        password = getpass.getpass("Input password for decrypting keyfile: ")
    with open(privkey_filepath, "rb") as f:
        return import_privKey_from_bytes(f.read(), password)


def import_privKey_from_bytes(encrypted_bytes, password):
    """Decrypt `_prv.enc` bytes (AES-sealed with SHA-256(password)) → PrivateKey."""
    decrypted = ecies.sym_decrypt(
        key=hashlib.sha256((password or "").encode()).digest(),
        cipher_text=encrypted_bytes,
    )
    return eth_keys.keys.PrivateKey(decrypted)


def save_pubkey(pubkey, pubkey_filepath):
    with open(pubkey_filepath, "wb") as f:
        f.write(pubkey.to_bytes())
    print(f"File written to {pubkey_filepath}")


def import_pubKey(pubkey_filepath):
    with open(pubkey_filepath, "rb") as f:
        return eth_keys.keys.PublicKey(f.read())


def save_addr(addr, addr_filepath):
    with open(addr_filepath, "wb") as f:
        f.write(addr.encode())
    print(f"Address written to {addr_filepath}: {addr}")


def import_addr(addr_filepath):
    with open(addr_filepath, "rb") as f:
        return f.read().decode()


def make_qr(data, image_path=None):
    qr = qrcode.QRCode(version=1, box_size=5, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    if image_path is not None:
        img.save(image_path)
    return img


def gen_save_keys_addr(basename_filepath, password=None, coin="Doge"):
    if os.path.isfile(basename_filepath + "_prv.enc"):
        privkey2save = import_privKey(basename_filepath + "_prv.enc", password)
    else:
        privkey2save = ecies.utils.generate_eth_key()
    pubkey2save = privkey2save.public_key
    save_privkey(privkey2save, basename_filepath + "_prv.enc", password=password)
    save_pubkey(pubkey2save, basename_filepath + "_pub.bin")
    version = pydoge.params.PUBKEY_ADDRESS if coin[0].lower() == "d" else 0x00
    addr2save = pydoge.pubtoaddr("04" + pubkey2save.to_bytes().hex(), version=version)
    save_addr(addr2save, basename_filepath + "_addr.bin")
    return make_qr(addr2save, basename_filepath + "_addr.png")
