"""
ladp/src/diagnostics — LADP diagnostic utilities
==================================================
Exposes the KNOR diagnostic and the loss-correctness correlation helper.
"""

from .knor import KNORDiagnostic, correlate_loss_with_error

__all__ = [
    "KNORDiagnostic",
    "correlate_loss_with_error",
]
