"""User-visible filesystem usage: exclude reserved blocks from availability."""


def disk_status(stat):
    used = stat.f_blocks - stat.f_bfree
    denominator = used + stat.f_bavail
    free_bytes = stat.f_bavail * stat.f_frsize
    percent = 100 * used / denominator if denominator else 100.0
    stop = (
        free_bytes < 40_000_000_000
        or denominator <= 0
        or 100 * used >= 95 * denominator
    )
    return free_bytes, percent, stop
