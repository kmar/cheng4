#include "rnd_shuf.h"

#ifdef _WIN32
#	include <Windows.h>
#endif

uint64_t seed_rnd()
{
#ifdef _WIN32
	static uint64_t base = 0;
	LARGE_INTEGER pc = {0};
	QueryPerformanceCounter(&pc);
	uint64_t res = pc.QuadPart;
	return ++base + res;
#else
	return 42;
#endif
}
