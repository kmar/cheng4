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
