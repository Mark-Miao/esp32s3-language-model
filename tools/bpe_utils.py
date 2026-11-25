"""Helpers to turn a ByteLevel BPE vocabulary into a device-friendly byte blob.

The tokenizer's vocab keys are GPT-2 byte-level strings ("Ġ" -> space, ...).
The firmware only needs to *decode*: id -> raw bytes. We export, for every id,
the concatenated bytes plus an offset table.
"""


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_B2U = bytes_to_unicode()
_U2B = {v: k for k, v in _B2U.items()}


def token_to_bytes(token: str) -> bytes:
    out = bytearray()
    for ch in token:
        if ch in _U2B:
            out.append(_U2B[ch])
        else:
            out.extend(ch.encode("utf-8"))
    return bytes(out)


def vocab_blob(tokenizer):
    """Return (offsets: list[int] of len V+1, blob: bytes)."""
    vocab = tokenizer.get_vocab()          # token -> id
    size = tokenizer.get_vocab_size()
    by_id = [None] * size
    for tok, i in vocab.items():
        if 0 <= i < size:
            by_id[i] = tok
    offsets, blob = [0], bytearray()
    for i in range(size):
        tok = by_id[i]
        if tok is None:
            offsets.append(len(blob))
            continue
        blob.extend(token_to_bytes(tok))
        offsets.append(len(blob))
    return offsets, bytes(blob)
