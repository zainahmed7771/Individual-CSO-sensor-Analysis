import numpy as np
from cso_spatial_drivers.fitting.stretched_exponential import fit_tail
def test_positive_bounded_fit():
 r=fit_tail(np.linspace(241,2000,40)); assert .01<=r['beta']<=2 and r['lambda_per_minute']>0 and r['n_tail']==40
