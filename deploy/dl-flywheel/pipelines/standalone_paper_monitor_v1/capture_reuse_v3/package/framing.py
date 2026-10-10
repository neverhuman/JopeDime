"""Pure physical JSONL framing: preserve bytes; never normalize line endings."""


def strict_lf_lines(wire):
    """Return original nonempty byte lines, requiring exactly LF framing."""
    if type(wire) is not bytes or not wire:
        raise ValueError("nonempty exact bytes required")
    if any(value < 0x20 and value != 0x0A for value in wire):
        raise ValueError("physical non-LF C0 control forbidden")
    try:
        text = wire.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("strict UTF-8 required") from error
    if any(separator in text for separator in ("\u0085", "\u2028", "\u2029")):
        raise ValueError("physical non-LF Unicode separator forbidden")
    parts = wire.split(b"\n")
    if parts[-1] != b"" or len(parts) < 2 or any(not line for line in parts[:-1]):
        raise ValueError("nonempty lines and exactly one trailing LF required")
    return tuple(parts[:-1])


def strict_lf_prefix(wire, count):
    """Return a physical prefix slice including its original final LF."""
    lines = strict_lf_lines(wire)
    if type(count) is not int or not 1 <= count <= len(lines):
        raise ValueError("positive available prefix count required")
    prefix = wire[:sum(len(line) + 1 for line in lines[:count])]
    if strict_lf_lines(prefix) != lines[:count]:
        raise ValueError("physical prefix framing mismatch")
    return prefix
