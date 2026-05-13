import tlsh
import numpy as np

def set_numpy_arr(h1):
    checksum    = 0
    lvalue      = 1
    q1value     = 2
    q2value     = 3
    narr1 = np.zeros(36, dtype=np.uint8)
    narr1[checksum]    = h1.checksum(0)
    narr1[lvalue]    = h1.lvalue
    narr1[q1value]    = h1.q1ratio
    narr1[q2value]    = h1.q2ratio
    for byte_idx in range(32):
        b0 = byte_idx * 4
        byte_val = 64 * h1.bucket_value(b0) + 16 * h1.bucket_value(b0+1) + 4 * h1.bucket_value(b0+2) + h1.bucket_value(b0+3)
        narr1[4 + byte_idx] = byte_val
    return(narr1)

def test_pynn(digest):
    h1 = tlsh.Tlsh()
    try:
        h1.fromTlshStr(digest)
    except:
        print("invalid digest: " + digest)
        sys.exit()
    # end try

    print(digest)

    numpy_arr1 = set_numpy_arr(h1)
    byte_arr2  = bytearray( [1] * 36 )
    h1.toByteArray(byte_arr2)

    err = 0
    for ni in range(0,36):
        if (byte_arr2[ni] != numpy_arr1[ni]):
            err = 1
            print("numpy: " + str(ni) + "=" + str(numpy_arr1[ni]) + " " )
            print("byte:  " + str(ni) + "=" + str(byte_arr2[ni])  + " " )
         # end if
    # end for
    if (err == 0):
        print("pass")

digest = "T11D9240993FD2E5A316D292A457865C5DF23C900B137CEA80FCEDE5DC6F5A03886F1A81"
test_pynn(digest)
digest = "T11454F100000000000000000000000000000000000000000000FFFFFFFFFFFFFFFFFF11"
test_pynn(digest)
