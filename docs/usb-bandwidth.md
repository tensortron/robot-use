# USB bandwidth: why three cameras can't all run at 720p

Short version: **all three cameras cannot stream at 720p or 1080p simultaneously.**
It is a USB 2.0 limit, not a camera limit — each camera does 1080p fine alone.

## The mechanism

All three cameras share **one USB 2.0 bus at 480 Mbps** (the cameras on one hub, as in
[HARDWARE_SETUP.md](HARDWARE_SETUP.md)). They are USB-2.0-only devices, so the empty
USB 3.0 bus cannot be used, and root-hub bandwidth is shared across all ports —
changing ports cannot help.

Webcams stream over **isochronous** transfers, which *reserve* guaranteed
bandwidth at stream start, held for the whole session whether or not data flows.
From this camera's own descriptors, its alternate settings are:

| alt | bytes/microframe | reserves |
|---|---|---|
| 1 | 128 | 8 Mbps |
| 2 | 256 | 16 Mbps |
| 3 | 800 | 51 Mbps |
| 4 | 1600 | 102 Mbps |
| 5 | 2400 | 154 Mbps |
| 6 | 3072 | **197 Mbps** |

(3072 bytes × 8000 microframes/s = 24.6 MB/s = 197 Mbps.)

```
1 camera at alt 6     197 Mbps   ✅
2 cameras             394 Mbps   ⚠️ already over the isochronous ceiling
3 cameras             590 Mbps   ❌

USB 2.0 bus                480 Mbps
Available to isochronous   384 Mbps   (spec caps periodic transfers at 80%)
```

At 720p/1080p there is only room for **two** reservations. The third camera's
request is refused (`-ENOSPC`) and it delivers **zero frames — silently, with no
error**. That silent failure is the dangerous part: it looks like a working
config until you inspect the data.

## Three consequences worth internalising

1. **MJPEG compression cannot save you.** The reservation is fixed at stream
   start from the worst case, not from how well a frame actually compresses. A
   black frame reserves as much as a detailed one.
2. **Port choice cannot save you.** 480 Mbps is per *bus*, shared across every
   port on that root hub. Moving the cameras to the "USB 3.0" hub changed
   nothing, because they are USB 2.0 devices.
3. **It is a step function, not a curve.** Bandwidth comes from a discrete ladder
   of six alt settings, so behaviour jumps rather than degrading smoothly.

## Measured matrix (all three cameras simultaneously, MJPG)

| Requested | Result |
|---|---|
| 1920×1080 | ❌ third starved |
| 1280×720 | ❌ third starved |
| 1024×576 | ✅ all OK |
| 800×600 | ❌ third starved |
| **640×480** | ❌ **third starved** |
| 640×360 | ✅ all OK |

**640×480 fails while 1024×576 succeeds** — reproducibly, verified from a clean
state. This is not noise and not a resolution ladder effect. The descriptors show
640×480 requests a *smaller* buffer (`dwMaxVideoFrameBufferSize` 614,989) than
1024×576 (1,180,237), so the firmware's per-mode bandwidth request is simply
**not monotonic in resolution**. Both negotiate MJPG. Treat the table as ground
truth; don't try to reason from resolution alone.

### Working mixed combinations

| Config | Result |
|---|---|
| **ego 1280×720 + both wrists 1024×576** | ✅ **chosen** — 27.6 / 26.8 / 25.9 fps sustained |
| ego 1024×576 + both wrists 1280×720 | ✅ works |
| ego 640×360 + both wrists 1280×720 | ✅ works |
| all three 1024×576 | ✅ 27.0 / 25.8 / 24.7 fps |

The budget is reallocatable: roughly "two cameras at 720p plus one lower". If
wrist detail later matters more than the ego view, the 720p slot can be moved to
a wrist with no hardware change. **You just cannot have all three.**

## Confirmed independent of hub power

The hub is bus-powered, which caused a separate class of fault (devices dropping
off USB -- see troubleshooting.md). The resolution ceiling is **not** that fault,
and a powered hub will not raise it.

Control experiment: the matrix above was re-run with **only the three cameras
connected and no arms at all**, removing essentially all competing current draw.
The results were identical, failure for failure:

| Resolution (all three) | With 4 arms also connected | Cameras only |
|---|---|---|
| 1920x1080 | third starved | third starved |
| 1280x720 | third starved | third starved |
| 1024x576 | all OK | all OK |
| 640x480 | third starved | third starved |

The 640x480 result is the discriminator on its own: current draw scales with
data rate, so if power were the cause the *lower* resolution would be the safer
one. It is not. Only a discrete ladder of bandwidth reservations explains a
lower resolution failing while a higher one succeeds.

Practical consequence: the powered hub will let four arms and three cameras
coexist, but the camera config stays ego 720p + wrists 1024x576. Raising that
needs USB 3.0 cameras or a CSI/MIPI camera, not more current.

## Frame rate is not a lever

Requesting 20/15/10 fps at 720p on all three still starves the third — the camera
negotiates 30 fps regardless. Its descriptors show MJPG 1280×720 has
`bFrameIntervalType 1` with exactly one interval, `333333` (30 fps). There is no
slower MJPG 720p mode.

(The 720p entry advertising 10/5 fps belongs to the *uncompressed* format, which
costs far more bandwidth, not less.)

## `uvcvideo quirks=128` — tried, did not help

`quirks` is a bitmask of workarounds on the `uvcvideo` module; `128` (`0x80`) is
`UVC_QUIRK_FIX_BANDWIDTH`. Normally the camera reports its bandwidth need during
PROBE/COMMIT negotiation and the driver reserves it; cheap cameras habitually
report an inflated worst case. This quirk tells the driver to ignore that and
compute the requirement itself, picking a smaller alt setting.

Exactly the right class of fix — it simply wasn't enough here. **Three 720p
streams still starve the third with the quirk enabled.**

It is currently active and worth persisting anyway (harmless, and it removes a
reboot-shaped variable):

```bash
echo 'options uvcvideo quirks=128' | sudo tee /etc/modprobe.d/uvcvideo.conf
```

> Caveat: whether the quirk is what enables `1024×576` on all three was never
> tested against a control (the matrix above was measured with it enabled). If
> you ever remove it, re-run the matrix.

## `fourcc: MJPG` is mandatory — the most dangerous default here

lerobot's `OpenCVCameraConfig.fourcc` defaults to `None` (auto-detect). Measured
on the ego camera at 1280×720:

| Setting | Negotiated | Frame rate |
|---|---|---|
| lerobot default (`fourcc=None`) | **YUYV** | **10 fps** |
| explicit `fourcc: MJPG` | MJPG | **30 fps** |

Without an explicit `fourcc`, OpenCV negotiates *uncompressed* YUYV and the
camera throttles to 10 fps to fit the bus. With three cameras it would starve
outright. **Every camera in every lerobot command must specify `fourcc: MJPG`**,
or you silently record a 10 fps dataset labelled 30.

## Getting three 720p streams (hardware only)

### Why the Orin's USB 3.2 Gen 2 ports do not already solve this

The board genuinely has SuperSpeed -- there are two root hubs:

```
Bus 001   480M      USB 2.0 pair (D+/D-)
Bus 002   10000M    SuperSpeed lanes (USB 3.2 Gen 2)
```

A USB 3.x port is **two independent buses sharing one connector**. A device
negotiates onto one of them according to its own capability, and every device
in this rig is USB-2.0-only:

```
1-4.1.2/3/4   480M   Innomaker cameras     -> bus 001
1-4.2.1/.4     12M   CH343 (followers)     -> bus 001 (full-speed)
1-4.3, 1-4.4   12M   CH343 (leaders)       -> bus 001 (full-speed)
2-3.1, 2-3.2  5000M  hubs only, no devices
```

Bus 002 carries nothing but hubs. The SuperSpeed lanes sit idle, and no choice
of port changes that: a USB 2.0 device in a USB 3.2 port still gets the same
shared 480 Mbps.

### Option 1: a USB 3.0 camera for the ego view (recommended)

The ego stream is the largest, so moving it off bus 001 frees the most for the
two wrist cameras that stay behind. Two cameras at 720p on bus 001 is already
proven to work, so the remaining pair should be comfortable.

**Verify it is genuinely SuperSpeed before trusting it.** Some cameras ship a
blue USB 3.0 connector but still enumerate as USB 2.0 devices, which would gain
nothing:

```bash
lsusb -t | grep -i video
```

Want it at **5000M or 10000M under Bus 002**. If it appears at `480M` under
bus 001, it is a USB 2.0 camera in a USB 3.0 shell -- return it.

### Option 2: a CSI/MIPI camera for the ego view

Bypasses USB entirely, freeing the whole USB budget. Cheapest of the routes and
the standard Jetson answer, at the cost of a fixed ribbon cable rather than a
USB lead.

### Not a fix

- **A powered hub** solves the *current* problem (see troubleshooting.md) but
  adds no bandwidth. Both purchases are needed; neither is redundant.
- **A USB 3.0 hub** does nothing on its own: USB 2.0 cameras behind it still
  negotiate onto bus 001.
- **A different Jetson port**: the Orin exposes a single USB 2.0 root hub, so
  every USB 2.0 device shares that one 480 Mbps pool.
