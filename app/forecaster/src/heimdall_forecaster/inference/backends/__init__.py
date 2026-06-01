"""Backend modules — importing this package triggers `@register` on each."""
from . import f0_seasonal_ar  # noqa: F401
from . import f1_lgbm        # noqa: F401
from . import f2_blr         # noqa: F401
from . import f3_deep_ensemble  # noqa: F401
from . import f5_f6_neural_process  # noqa: F401
from . import f7_patch_tst    # noqa: F401
from . import f8_patch_tst    # noqa: F401
from . import f9_timesfm      # noqa: F401
from . import f10_chronos_bolt  # noqa: F401
from . import f11_pricefm    # noqa: F401
from . import ar1_fallback    # noqa: F401
from . import f12_multizone   # noqa: F401
from . import f13_multivariate  # noqa: F401
from . import f3_lite_deepar  # noqa: F401
from . import f4_mc_dropout   # noqa: F401
