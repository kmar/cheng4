#include "memmap.h"

#ifdef _WIN32
#	include <Windows.h>
#endif

const uint8_t *memory_mapped_file::map(const char *fn)
{
	unmap();
	void *res = nullptr;

	handles[0] = handles[1] = nullptr;

#ifdef _WIN32
	HANDLE handle = CreateFileA(fn, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

	if (handle)
	{
		HANDLE mapping = CreateFileMappingA(handle, NULL, PAGE_READONLY, 0, 0, NULL);

		handles[0] = handle;
		handles[1] = mapping;

		if (mapping)
			res = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);

		if (res)
		{
			DWORD szhi = 0;
			DWORD szlo = GetFileSize(handle, &szhi);
			mapped_size = (int64_t)szlo + ((int64_t)szhi << 32);
		}
	}

	mapped = res;
#endif

	return static_cast<const uint8_t *>(res);
}

void memory_mapped_file::unmap()
{
#ifdef _WIN32
	if (mapped)
	{
		UnmapViewOfFile(mapped);
		mapped = nullptr;
		mapped_size = 0;
	}

	if (handles[1])
		CloseHandle(handles[1]);
	if (handles[0])
		CloseHandle(handles[0]);

	handles[0] = handles[1] = nullptr;
#endif
}
