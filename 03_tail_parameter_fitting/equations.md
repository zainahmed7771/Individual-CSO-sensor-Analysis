# Tail model

For event duration `T` and threshold `u=240` minutes, the fitted conditional survival for `T>u` is based on

`S(t) = exp[-(lambda*t)^beta]`.

The conditional log-likelihood subtracts the survival contribution at `u`. The optimizer bounds are `log(lambda) in [-50,20]` and `beta in [0.01,2]`. Lower beta means a heavier/more persistent fitted tail. Lambda is an inverse timescale in this parameterisation. Exact boundary fits are diagnostics, not values to clip silently into transformed regressions.
