#include "cucim/util/file.h"

#include <sys/stat.h>

namespace cucim::util
{

bool file_exists(const char* path)
{
    struct stat st_buff;
    return stat(path, &st_buff) == 0;
}

} // namespace cucim::util