"""Decoupled deixis — resolve what the user points at, off the audio hot path.

The Week-4 finding this productionizes: the realtime audio model is the wrong reader.
A small vision model (Flash-Lite) given the cursor tile + AX hints resolves "this/that"
referents better (~88% vs ~80% bench) and off the hot path, so the live model receives a
pre-digested ``pointer=`` annotation instead of doing the reading itself.
"""

from .resolver import PointerContext, PointerReferentResolver

__all__ = ["PointerContext", "PointerReferentResolver"]
