"""Crustal magnetic anomaly grids synthesised from a spherical-harmonic field model.

No gridded global magnetic anomaly map (EMAG2, WDMAM) was reachable from the build
machine, but the NOAA Enhanced Magnetic Model 2017 was, as GeographicLib's binary
coefficient file. EMM2017 carries the crustal field to spherical-harmonic degree 790
(~50 km wavelength) and was built from EMAG2 plus satellite data, so synthesising it
recovers an EMAG2-like map at the resolution that matters for regional prospectivity.

Layout of a GeographicLib "WMMF-2" file: an 8-byte model id, then blocks of
int32 N, int32 M, (M+1)(2N-M+2)/2 float64 C (cosine) coefficients ordered by m then n,
and M(2N-M+1)/2 float64 S (sine) coefficients ordered the same way from m = 1. EMM2017
holds 18 main-field epochs (2000..2017, degree 15), a secular-variation block, and one
crustal block of degree 790. Coefficients are Schmidt semi-normalised, in nT.

The total-field anomaly is the crustal field projected onto the main-field direction,
which is what an aeromagnetic survey measures. Derived layers follow the gravity stack:
a 100 km regional, the residual, its horizontal gradient, vertical derivative and tilt.
"""

from __future__ import annotations

import struct

import numpy as np

from .rasters import GeoGrid

A_REF = 6371200.0  # m, reference radius of geomagnetic models


def read_wmmf(path: str) -> tuple[str, list[np.ndarray]]:
    """Return (model id, [coefficient arrays (2, N+1, N+1) per block]) with
    coeffs[0, n, m] = g_nm and coeffs[1, n, m] = h_nm."""
    b = open(path, "rb").read()
    model_id, off, blocks = b[:8].decode(), 8, []
    while off < len(b):
        N, M = struct.unpack("<ii", b[off:off + 8])
        off += 8
        nc, ns = (M + 1) * (2 * N - M + 2) // 2, M * (2 * N - M + 1) // 2
        C = np.frombuffer(b, "<f8", nc, off)
        off += 8 * nc
        S = np.frombuffer(b, "<f8", ns, off)
        off += 8 * ns
        k = np.zeros((2, N + 1, N + 1))
        ic = js = 0
        for m in range(M + 1):
            k[0, m:, m] = C[ic:ic + N + 1 - m]
            ic += N + 1 - m
            if m:
                k[1, m:, m] = S[js:js + N + 1 - m]
                js += N + 1 - m
        blocks.append(k)
    return model_id, blocks


def _field(coeffs, lmax_grid: int, altitude_m: float):
    import pyshtools as sh
    c = sh.SHMagCoeffs.from_array(coeffs, r0=A_REF, normalization="schmidt", csphase=1)
    g = c.expand(a=A_REF + altitude_m, lmax=lmax_grid, sampling=2, extend=False)
    return g.rad.data, g.theta.data, g.phi.data, g.rad.lats(), g.rad.lons()


def total_field_anomaly(path: str, altitude_m: float = 0.0, lmax: int | None = None) -> GeoGrid:
    """Crustal total-field anomaly (nT) on a ~(180 / lmax) degree grid."""
    _, blocks = read_wmmf(path)
    crust = blocks[-1]
    main = blocks[-3]  # latest main-field epoch (the block after it is secular variation)
    L = crust.shape[1] - 1 if lmax is None else lmax
    crust = crust[:, :L + 1, :L + 1]
    br, bt, bp, lats, lons = _field(crust, L, altitude_m)
    mr, mt, mp, *_ = _field(main, L, altitude_m)
    norm = np.sqrt(mr ** 2 + mt ** 2 + mp ** 2)
    dT = (br * mr + bt * mt + bp * mp) / norm
    return GeoGrid(lats, lons, dT, "mag_tmi")


def magnetic_stack(tmi: GeoGrid) -> dict[str, GeoGrid]:
    from . import geophysics as gp
    reg = gp.gaussian_lowpass(tmi, 100.0)
    resid = tmi.with_values(tmi.values - reg.values, "mag_resid100")
    vd = gp.vertical_derivative(resid)
    hg = gp.hgm(resid)
    return {
        "mag_tmi": tmi,
        "mag_tmi_abs": tmi.with_values(np.abs(tmi.values)),
        "mag_reg100": reg,
        "mag_resid100": resid,
        "mag_hgm": hg,
        "mag_vd": vd,
        "mag_tilt": gp.tilt(resid),
        # Analytic-signal amplitude: peaks over magnetic sources whatever the field
        # inclination, so no reduction to the pole is needed.
        "mag_analytic_signal": resid.with_values(np.hypot(hg.values, vd.values)),
    }
