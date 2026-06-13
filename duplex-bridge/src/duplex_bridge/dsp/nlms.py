"""Normalized LMS (NLMS) adaptive echo canceller (pure numpy, stateful).

The :class:`NlmsCanceller` estimates the linear echo path between a far-end
reference signal (what was played out of the speaker) and the near-end mic
signal (near-end speech plus an echo of the far-end). It subtracts the
estimated echo, returning the residual.

State (adaptive filter taps and the far-end tap-delay history) persists across
``process`` calls, so the canceller can be fed small frames (e.g. 10 ms) in
real time without resetting its convergence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class NlmsCanceller:
    """Stateful normalized-LMS acoustic echo canceller.

    Update rule per sample ``n``::

        x      = [far[n], far[n-1], ..., far[n-L+1]]   (tap-delay line)
        e[n]   = near[n] - w . x
        w     += (mu / (x . x + eps)) * e[n] * x

    where ``w`` are the adaptive filter taps (length ``filter_len``), ``mu`` is
    the step size and ``eps`` regularizes the normalization to avoid divide-by-zero.
    """

    def __init__(
        self,
        filter_len: int = 256,
        mu: float = 0.5,
        eps: float = 1e-3,
        dtd_threshold: float = 0.5,
        dtd_hangover: int = 240,
    ) -> None:
        if filter_len < 1:
            raise ValueError("filter_len must be >= 1")
        if not (0.0 < mu <= 2.0):
            raise ValueError("mu must be in (0, 2]")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        if dtd_threshold <= 0.0:
            raise ValueError("dtd_threshold must be > 0")
        if dtd_hangover < 0:
            raise ValueError("dtd_hangover must be >= 0")

        self.filter_len = int(filter_len)
        self.mu = float(mu)
        self.eps = float(eps)
        # Geigel double-talk detector threshold: double-talk is flagged when the
        # near-end sample magnitude exceeds dtd_threshold * (recent peak |far|).
        # Loud near-end (relative to the far-end driving the echo) is the
        # signature of double-talk, during which adapting would chase — and thus
        # suppress — near-end speech. Adaptation is frozen while flagged.
        self.dtd_threshold = float(dtd_threshold)
        # Hold-over: keep adaptation frozen for this many samples after the last
        # double-talk flag, so brief near-end zero-crossings don't let the filter
        # resume adapting (and diverge) mid double-talk. 240 ≈ 15 ms @ 16 kHz.
        self.dtd_hangover = int(dtd_hangover)
        self._hangover_left = 0

        # Adaptive filter taps, newest far-end sample aligns with w[0].
        self._w: NDArray[np.float64] = np.zeros(self.filter_len, dtype=np.float64)
        # Tap-delay line of past far-end samples: history[0] is most recent.
        self._history: NDArray[np.float64] = np.zeros(self.filter_len, dtype=np.float64)
        # Bound to keep tap energy from diverging on pathological input.
        self._max_tap_norm = 1.0e6

    def reset(self) -> None:
        """Clear adaptive taps, far-end history, and double-talk state."""
        self._w.fill(0.0)
        self._history.fill(0.0)
        self._hangover_left = 0

    @property
    def taps(self) -> NDArray[np.float64]:
        """A copy of the current adaptive filter taps."""
        taps: NDArray[np.float64] = self._w.copy()
        return taps

    def process(
        self,
        near: NDArray[np.float64],
        far_ref: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Cancel the echo of ``far_ref`` present in ``near``.

        Args:
            near: Mic signal (near-end speech + echo of ``far_ref``).
            far_ref: Far-end reference (what was played). Same length as ``near``.

        Returns:
            The residual (echo-cancelled near signal), same length as ``near``.
        """
        near_arr = np.asarray(near, dtype=np.float64).reshape(-1)
        far_arr = np.asarray(far_ref, dtype=np.float64).reshape(-1)
        if near_arr.shape[0] != far_arr.shape[0]:
            raise ValueError("near and far_ref must have the same length")

        n_samples = near_arr.shape[0]
        out = np.empty(n_samples, dtype=np.float64)

        w = self._w
        hist = self._history
        mu = self.mu
        eps = self.eps
        max_norm = self._max_tap_norm
        dtd_threshold = self.dtd_threshold

        for i in range(n_samples):
            # Shift the newest far-end sample into the front of the delay line.
            hist[1:] = hist[:-1]
            hist[0] = far_arr[i]

            near_i = float(near_arr[i])
            y = float(np.dot(w, hist))  # estimated echo
            e = near_i - y  # residual / error
            out[i] = e

            # Geigel double-talk detector: if the near-end magnitude is large
            # relative to the recent far-end peak, near-end speech is present.
            # Freeze adaptation (with hold-over) so we don't chase — and thereby
            # cancel — near-end speech.
            far_peak = float(np.max(np.abs(hist)))
            if far_peak > 0.0 and abs(near_i) > dtd_threshold * far_peak:
                self._hangover_left = self.dtd_hangover

            if self._hangover_left > 0:
                self._hangover_left -= 1
            else:
                norm = float(np.dot(hist, hist)) + eps
                w += (mu * e / norm) * hist

                # Guard against tap divergence on pathological / unstable input.
                tap_norm = float(np.dot(w, w))
                if tap_norm > max_norm * max_norm:
                    w *= max_norm / np.sqrt(tap_norm)

        return out
