"""Factory helpers for paper ablation variants of CoTracker3 Online.

The main source files keep the original Base implementation and the full
A+B+C implementation separate. This module provides one stable entry point for
benchmarking paper variants without repeatedly editing model source files.
"""

from cotracker.models.core.cotracker.blocks import (
    BasicEncoder,
    BaseEncoder,
    ConservativeBaseEncoder,
    RigidDescriptorEncoder,
    StandaloneLRGConservativeFPNEncoder,
    StandaloneLRGMRFPNEncoder,
)
from cotracker.models.core.cotracker.cotracker3_online import (
    CoTrackerThreeOnline as BaseOnline,
)
from cotracker.models.core.cotracker.full import (
    CoTrackerThreeOnline as FullOnline,
)


VARIANT_ALIASES = {
    "base": "base",
    "baseline": "base",
    "base_a": "base_a",
    "a": "base_a",
    "base+a": "base_a",
    "base_b": "base_b",
    "b": "base_b",
    "base+b": "base_b",
    "base_b_conservative": "base_b_conservative",
    "b_conservative": "base_b_conservative",
    "ab": "ab_conservative",
    "a_b": "ab_conservative",
    "a+b": "ab_conservative",
    "base_ab": "ab_conservative",
    "ab_mrfpn": "ab_mrfpn",
    "a+b_mrfpn": "ab_mrfpn",
    "ab_conservative": "ab_conservative",
    "a+b_conservative": "ab_conservative",
    "base_c": "base_c",
    "c": "base_c",
    "base+c": "base_c",
    "abc": "abc",
    "a+b+c": "abc",
    "full": "abc",
    "abc_full": "abc",
}


def normalize_online_variant(variant):
    key = (variant or "base").lower().replace("-", "_")
    if key not in VARIANT_ALIASES:
        valid = ", ".join(sorted(VARIANT_ALIASES))
        raise ValueError(f"Unknown online variant '{variant}'. Valid variants: {valid}")
    return VARIANT_ALIASES[key]


def _replace_fnet(model, fnet):
    model.fnet = fnet
    return model


def build_cotracker_three_online_variant(variant="base", **kwargs):
    """Build the requested paper variant.

    Variants:
    - base: original CoTracker3 Online.
    - base_a: Base + LRG-Encoder.
    - base_b: Base + MR-FPN front-end, using the recovered routing FPN.
    - base_b_conservative: Base + conservative routing FPN.
    - ab: LRG-Encoder + conservative FPN, matching the saved AB.pth.
    - ab_mrfpn: LRG-Encoder + recovered routing MR-FPN.
    - ab_conservative: explicit alias for the saved AB.pth architecture.
    - base_c: Base front-end + RAMP-Former.
    - abc/full: LRG-Encoder + MR-FPN + RAMP-Former.
    """

    variant = normalize_online_variant(variant)

    if variant == "base":
        return BaseOnline(**kwargs)

    if variant == "abc":
        return FullOnline(**kwargs)

    if variant == "base_c":
        model = FullOnline(**kwargs)
        return _replace_fnet(
            model,
            BasicEncoder(input_dim=3, output_dim=model.latent_dim, stride=model.stride),
        )

    if variant == "base_a":
        model = BaseOnline(**kwargs)
        return _replace_fnet(
            model,
            RigidDescriptorEncoder(
                input_dim=3, output_dim=model.latent_dim, stride=model.stride
            ),
        )

    if variant == "base_b":
        model = BaseOnline(**kwargs)
        return _replace_fnet(
            model,
            BaseEncoder(input_dim=3, output_dim=model.latent_dim, stride=model.stride),
        )

    if variant == "base_b_conservative":
        model = BaseOnline(**kwargs)
        return _replace_fnet(
            model,
            ConservativeBaseEncoder(
                input_dim=3, output_dim=model.latent_dim, stride=model.stride
            ),
        )

    if variant == "ab_mrfpn":
        model = BaseOnline(**kwargs)
        return _replace_fnet(
            model,
            StandaloneLRGMRFPNEncoder(
                input_dim=3, output_dim=model.latent_dim, stride=model.stride
            ),
        )

    if variant == "ab_conservative":
        model = BaseOnline(**kwargs)
        return _replace_fnet(
            model,
            StandaloneLRGConservativeFPNEncoder(
                input_dim=3, output_dim=model.latent_dim, stride=model.stride
            ),
        )

    raise AssertionError(f"Unhandled normalized variant: {variant}")
