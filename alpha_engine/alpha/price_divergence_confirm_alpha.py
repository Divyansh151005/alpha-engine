"""Option B — second alpha: price return must agree with the funding-contrarian thesis.

Use together with :class:`~alpha_engine.alpha.funding_contrarian_alpha.FundingContrarianAlphaStrategy`
and a shared ``funding`` instance passed into ``FundingPriceConfirmContrarianAlphaStrategy``-compatible
``__init__`` (via this subclass). :class:`~alpha_engine.alpha.alpha_engine.AlphaEngine` applies an
AND gate: the raw funding leg is dropped unless this strategy also emits on the same bar.

Signal generation (funding validity, price confirmation) is inherited from
:class:`~alpha_engine.alpha.funding_price_confirm_alpha.FundingPriceConfirmContrarianAlphaStrategy`.
"""

from __future__ import annotations

from alpha_engine.alpha.funding_price_confirm_alpha import FundingPriceConfirmContrarianAlphaStrategy


class PriceDivergenceContrarianConfirmAlphaStrategy(FundingPriceConfirmContrarianAlphaStrategy):
    """Price divergence confirmation for funding contrarian (shared funding buffer required)."""
