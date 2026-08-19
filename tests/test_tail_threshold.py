from cso_spatial_drivers.fitting.stretched_exponential import tail_values
def test_tail_is_strictly_greater_than_240():
 assert tail_values([239,240,240.0001]).tolist()==[240.0001]
