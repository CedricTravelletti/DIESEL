from libc.math cimport sin, cos, asin, sqrt, atan2
import numpy as np
cimport numpy as np

cdef double NAN = <double> np.nan

   
## Equivalent to 3.1415927 / 180
cdef double PI_RATIO = 0.017453293

cdef double deg2rad(double deg):
    cdef double rad = deg * PI_RATIO
    return rad
           
def haversine(double lat1, double lon1, double lat2, double lon2):
    cdef double rlon1 = deg2rad(lon1)
    cdef double rlon2 = deg2rad(lon2)
    cdef double rlat1 = deg2rad(lat1)
    cdef double rlat2 = deg2rad(lat2)
                                   
    cdef double dlon = rlon2 - rlon1
    cdef double dlat = rlat2 - rlat1

    cdef double a = (
            cos(rlat2) * sin(dlon))**2 + (cos(rlat1) * sin(rlat2)
                    - sin(rlat1) * cos(rlat2) * cos(dlon))**2
    cdef double b = sin(rlat1) * sin(rlat2) + cos(rlat1) * cos(rlat2) * cos(dlon)

    cdef double c = atan2(sqrt(a), b)
    cdef double km = 6371 * c
    return km
