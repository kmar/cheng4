/*
You can use this program under the terms of either the following zlib-compatible license
or as public domain (where applicable)

  Copyright (C) 2012-2015, 2020-2021, 2023-2024 Martin Sedlak

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgement in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

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

	if (handle != INVALID_HANDLE_VALUE)
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
