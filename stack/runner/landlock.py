"""Layer 3 of the program sandbox: Landlock, the kernel's unprivileged access control (Linux 5.13+).

Layers 1 and 2 (static_check, the audit hook in child.py) only see Python. A full review on 2026-10-03 escaped
them twice: through `runtime.os.sys.modules["cv2"]`, then through strings that Python evaluates
(`typing.get_type_hints` on a string annotation, pickle inside `numpy.load`). OpenCV then wrote files and
opened URLs in C, where no audit hook looks. Landlock is enforced by the kernel on every syscall of the
process and its children, cannot be lifted once applied, and needs no privileges:

- the file system becomes read-only, and readable only beneath the paths given (the Python install, the
  repository, the shared libraries, a few /dev nodes); nothing can be created, written, removed or renamed;
- with ABI 4 (Linux 6.7+) TCP bind and connect are refused, so robotd's API, the MCP server and the network
  are out of reach. (UDP and Unix sockets are not covered by Landlock yet; the audit hook refuses Python
  sockets, and robotd speaks TCP only.)

Inherited fds (the runner's pipes) keep working: Landlock checks opening, not using, a file.
"""

from __future__ import annotations

import ctypes
import os
import platform

_SYS = {"x86_64": (444, 445, 446), "aarch64": (444, 445, 446)}     # create_ruleset, add_rule, restrict_self
PR_SET_NO_NEW_PRIVS = 38
CREATE_RULESET_VERSION = 1
RULE_PATH_BENEATH = 1

FS_EXECUTE, FS_WRITE_FILE, FS_READ_FILE, FS_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
FS_REMOVE_DIR, FS_REMOVE_FILE, FS_MAKE_CHAR, FS_MAKE_DIR = 1 << 4, 1 << 5, 1 << 6, 1 << 7
FS_MAKE_REG, FS_MAKE_SOCK, FS_MAKE_FIFO, FS_MAKE_BLOCK, FS_MAKE_SYM = 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12
FS_REFER, FS_TRUNCATE = 1 << 13, 1 << 14                               # ABI 2, ABI 3
NET_BIND_TCP, NET_CONNECT_TCP = 1 << 0, 1 << 1                         # ABI 4

READ = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class LandlockUnavailable(RuntimeError):
    pass


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def abi_version() -> int:
    """The kernel's Landlock ABI (0 if unsupported or disabled)."""
    nums = _SYS.get(platform.machine())
    if nums is None:
        return 0
    r = _libc().syscall(nums[0], None, ctypes.c_size_t(0), ctypes.c_uint32(CREATE_RULESET_VERSION))
    return int(r) if r > 0 else 0


def restrict(read_paths, min_abi: int = 4, use_abi: int | None = None) -> int:
    """Apply to this process (and its future children): read-only beneath ``read_paths``, nothing else, and with
    ABI 4 no TCP. Raises LandlockUnavailable if the kernel's ABI is below ``min_abi``. Returns the ABI used.

    Below ABI 4 (min_abi < 4: the simulated rig on an older kernel, sprint 2026-10-13 D14) only what that ABI
    handles is restricted: the file system from ABI 1 (renames across directories from 2, truncation from 3), TCP
    not at all, which leaves sockets to the audit hook. ``use_abi`` caps the features used (tests)."""
    abi = abi_version()
    if abi < min_abi:
        raise LandlockUnavailable(f"Landlock ABI {abi} < {min_abi} (needs Linux 6.7+ with landlock in the LSM list)")
    if use_abi is not None:
        abi = min(abi, use_abi)
    libc = _libc()
    create, add, restrict_self = _SYS[platform.machine()]
    fs = (FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_READ_DIR | FS_REMOVE_DIR | FS_REMOVE_FILE | FS_MAKE_CHAR
          | FS_MAKE_DIR | FS_MAKE_REG | FS_MAKE_SOCK | FS_MAKE_FIFO | FS_MAKE_BLOCK | FS_MAKE_SYM)
    fs |= (FS_REFER if abi >= 2 else 0) | (FS_TRUNCATE if abi >= 3 else 0)
    attr = _RulesetAttr(fs, (NET_BIND_TCP | NET_CONNECT_TCP) if abi >= 4 else 0)
    rfd = libc.syscall(create, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0))
    if rfd < 0:
        raise OSError(ctypes.get_errno(), f"landlock_create_ruleset: {os.strerror(ctypes.get_errno())}")
    try:
        for p in read_paths:
            if not os.path.exists(p):
                continue
            fd = os.open(p, os.O_PATH | os.O_CLOEXEC)
            try:
                access = READ if os.path.isdir(p) else (FS_EXECUTE | FS_READ_FILE)
                rule = _PathBeneath(access, fd)
                if libc.syscall(add, ctypes.c_int(rfd), ctypes.c_int(RULE_PATH_BENEATH), ctypes.byref(rule),
                                ctypes.c_uint32(0)) < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule {p}: {os.strerror(ctypes.get_errno())}")
            finally:
                os.close(fd)
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS)")
        if libc.syscall(restrict_self, ctypes.c_int(rfd), ctypes.c_uint32(0)) < 0:
            raise OSError(ctypes.get_errno(), f"landlock_restrict_self: {os.strerror(ctypes.get_errno())}")
    finally:
        os.close(rfd)
    return abi
