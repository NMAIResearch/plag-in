"""Registered native ABI profiles accepted by the embedded runtime.

Each profile binds the Python FFI layout to exact native component bytes.
Adding another native build requires a new reviewed profile and regression
coverage. A matching filename or user-supplied version label is insufficient.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeAbiProfile:
    profile_id: str
    upstream_identity: str
    library_sha256: str
    ggml_base_library_sha256: str
    ggml_library_sha256: str
    backend_library_sha256: tuple[str, ...]

    def matches(
        self,
        *,
        upstream_identity: str,
        library_sha256: str,
        ggml_base_library_sha256: str,
        ggml_library_sha256: str,
        backend_library_sha256: tuple[str, ...],
    ) -> bool:
        return (
            self.upstream_identity == upstream_identity
            and self.library_sha256 == library_sha256
            and self.ggml_base_library_sha256 == ggml_base_library_sha256
            and self.ggml_library_sha256 == ggml_library_sha256
            and self.backend_library_sha256 == backend_library_sha256
        )


REGISTERED_NATIVE_ABI_PROFILES = (
    NativeAbiProfile(
        profile_id="linux-x86_64-ollama-v0.32.13-llama-b10380",
        upstream_identity=(
            "Ollama v0.32.13 payload; llama.cpp b10380; "
            "Ollama compatibility patches"
        ),
        library_sha256=(
            "ae41185f48c9eb42b46db1d531e35b73"
            "fb1815787a83b6521b9372e6a4517fa3"
        ),
        ggml_base_library_sha256=(
            "3e9d33d6f2a3df69c23556b2b4f219b"
            "dc33902b03a45f3f3e65f275ed9381b2e"
        ),
        ggml_library_sha256=(
            "959e2fe38c212fec3596c56947ce2ab09"
            "c9ed8ed14e9fa667f25b5a409b86804"
        ),
        backend_library_sha256=(
            "4db625785dc67261632863708504efa56"
            "f3be5b16caa2399330b31fd184490cc",
            "2d2c27f58a2725d35b76347308fb7070"
            "fa9e9759757b3f6f2a641eded2547982",
        ),
    ),
)


def identify_native_abi_profile(
    *,
    upstream_identity: str,
    library_sha256: str,
    ggml_base_library_sha256: str,
    ggml_library_sha256: str,
    backend_library_sha256: tuple[str, ...],
    profiles: tuple[NativeAbiProfile, ...] | None = None,
) -> NativeAbiProfile | None:
    """Return the exact registered profile for one verified native bundle.

    The registry is read when the lookup runs rather than when this function
    is defined, so the set of registered profiles has one live definition
    and callers cannot hold a stale copy of it.
    """
    if profiles is None:
        profiles = REGISTERED_NATIVE_ABI_PROFILES
    for profile in profiles:
        if profile.matches(
            upstream_identity=upstream_identity,
            library_sha256=library_sha256,
            ggml_base_library_sha256=ggml_base_library_sha256,
            ggml_library_sha256=ggml_library_sha256,
            backend_library_sha256=backend_library_sha256,
        ):
            return profile
    return None
