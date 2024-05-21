#include "wiregrad.h"


namespace wiregrad::cuda {

bool is_available()
{
#if defined(WIREGRAD_CUDA)
    return true;
#else
    return false;
#endif
}

} // namespace wiregrad::cuda



