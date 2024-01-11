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

#pragma once

#include <cstdint>

struct memory_mapped_file
{
	// returns null on error
	const uint8_t *map(const char *fn);
	void unmap();

	~memory_mapped_file()
	{
		unmap();
	}

	int64_t size() const
	{
		return mapped_size;
	}

private:
	int64_t mapped_size = 0;
	const void *mapped = nullptr;
	// OS-specific handles
	void *handles[2] = {nullptr, nullptr};
};
