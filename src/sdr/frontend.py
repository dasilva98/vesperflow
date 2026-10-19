import numpy as np

FS = 1.024e6
N_MS = int(FS * 0.001)  # 1024 complex samples per 1 ms epoch


def unpack_vesper_iq(filepath: str, total_ms: int = 480):
    """
    Unpacks Vesper .dat snapshot files (1-bit Complex I/Q at 1.024 MHz).
    Assumes gps_parser.py has stripped the header and applied the 32-bit half-word swap.
    Applies the required 16-bit byte swap and LSB-first bit unpacking.
    """
    num_bytes = int(FS * (total_ms + 2) * 1e-3) // 4
    with open(filepath, "rb") as f:
        raw_bytes = f.read(num_bytes)

    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    if buf.size < N_MS:
        return np.array([], dtype=np.complex64), 0

    # Vesper ACLYS hardware 16-bit adjacent byte swap
    buf = buf[: (buf.size // 2) * 2].reshape(-1, 2)[:, ::-1].ravel()

    bits = np.unpackbits(buf, bitorder="little")
    s = 1.0 - 2.0 * bits.astype(np.float32)
    samples = (s[0::2] + 1j * s[1::2]).astype(np.complex64)

    actual_ms = min(total_ms, (len(samples) // N_MS) - 1)
    if actual_ms <= 0:
        return np.array([], dtype=np.complex64), 0

    # Skip 1st millisecond to avoid RF front-end power-on transient and remove DC offset
    seg = samples[N_MS : (actual_ms + 1) * N_MS]
    return seg - np.mean(seg), actual_ms
