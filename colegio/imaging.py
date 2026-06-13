"""imaging — the image bit-codec for image quipus (0x03).

Pure numpy/PIL: encode an image to a bit array at a given bit-depth and color,
and decode an image-quipu body back to a pixel array. No chain dependency.
"""

import numpy as np
from PIL import Image


def grey_imgarr(imgarr):
    return imgarr[:, :, :3].mean(axis=2).astype("uint8")


def message_2_bit_array(message, mode=None):
    """str / bytes / hex-str → uint8 bit-array (MSB first)."""
    if isinstance(message, bytes):
        hex_str = message.hex()
    elif isinstance(message, str):
        hex_str = message if mode in ("hex", "hexstring") else message.encode().hex()
    else:
        raise TypeError("message must be bytes or str")
    num = int("0x" + hex_str, base=16)
    bit_len = ((len(hex_str) + 1) // 2) * 8
    bin_str = bin(num)[2:]
    bits = [0] * (bit_len - len(bin_str)) + [int(b) for b in bin_str]
    return np.array(bits, dtype="uint8")


def bit_array_2_byte_str(bit_array):
    bin_str = "0b" + "".join(str(b) for b in bit_array)
    return int(bin_str, base=2).to_bytes(len(bit_array) // 8, "big")


def bit_array_2_hex_str(bit_array):
    return bit_array_2_byte_str(bit_array).hex()


def bit_array_2_str(bit_array, encoding="utf-8"):
    return bit_array_2_byte_str(bit_array).decode(encoding)


def int2bitarray(x, bit=8):
    return message_2_bit_array(hex(x)[2:], mode="hex")[:bit]


def bitarray2int(b_arr):
    ln = b_arr.shape[0]
    scales = (2 ** np.arange(7, -1, -1))[:ln]
    return (b_arr * scales).sum()


def imgarr2bitarray(imgarr, bit=8):
    return np.array([int2bitarray(it, bit) for it in imgarr.reshape(-1)]).reshape(-1)


def bitarray2imgarr(barrs, imgshape=(16, 16), bit=2, color=1):
    flat = barrs.reshape(-1)
    ints = [bitarray2int(flat[i:i + bit]) for i in range(0, len(flat), bit)]
    return np.array(ints).reshape(*imgshape, color).astype("uint8")


class bitimage:
    """Resize an image to dims, encode to a bit array at given bit-depth and color."""

    def __init__(self, imgpath, dims=(16, 16), bit=2, color=1):
        self.color = color
        self.bit = bit
        self.dims = list(dims)
        self.img_og = Image.open(imgpath)
        self.img_resize = self.img_og.resize(dims)
        self.grey = grey_imgarr(np.array(self.img_resize))
        self.img_grey = Image.fromarray(self.grey)
        self.bitarray = imgarr2bitarray(self.grey, bit)
        self.bitarray_color = imgarr2bitarray(np.array(self.img_resize)[:, :, :color], bit)
        self.newimg = Image.fromarray(
            bitarray2imgarr(self.bitarray, imgshape=dims[::-1], bit=bit, color=1).squeeze())
        self.newimg_color = Image.fromarray(
            bitarray2imgarr(self.bitarray_color, imgshape=dims[::-1], bit=bit, color=3).squeeze())
        self.bytestring = bit_array_2_byte_str(self.bitarray)
        self.bytestring_color = bit_array_2_byte_str(self.bitarray_color)


def read_image_data(hex_header, image_bytes):
    """Decode an image-quipu given the header hex and the body bytes."""
    C = {0: 1, 1: 3}[int(hex_header[12:14], 16)]
    L = int(hex_header[14:18], 16)
    W = int(hex_header[18:22], 16)
    B = int(hex_header[22:24], 16)
    bits = message_2_bit_array(image_bytes, mode=None)
    return bitarray2imgarr(bits, imgshape=(W, L), bit=B, color=C).squeeze()
