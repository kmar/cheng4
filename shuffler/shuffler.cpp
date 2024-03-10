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

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <vector>
#include <string>

#include <random>
#include <chrono>
#include <utility>

#include <fstream>
#include <iostream>

// matches cheng's entry size
size_t entry_size = 38;

// min/max entries per chunk
size_t min_entries = 50'000'000;
size_t max_entries = 100'000'000;

std::string in_filename;
std::string out_filename;

struct Plan
{
	uint64_t source_file_offset;
	uint64_t num_entries;
};

std::vector<Plan> plan;

int parse_args(int argc, const char **argv)
{
	int file_name_idx = 0;

	for (int i=1; i<argc; i++)
	{
		std::string arg = argv[i];

		// ugh... no starts_with for string
		if (arg.length() > 2 && arg[0] == '-' && arg[1] == '-')
		{
			if (i+1 >= argc)
			{
				std::cerr << "no argument after " << arg << std::endl;
				return 2;
			}

			// entry size
			if (arg == "--entry")
			{
				entry_size = strtol(argv[i+1], nullptr, 10);
				i++;
			}

			if (arg == "--min")
			{
				min_entries = strtol(argv[i+1], nullptr, 10);
				i++;
			}

			if (arg == "--max")
			{
				max_entries = strtol(argv[i+1], nullptr, 10);
				i++;
			}

			continue;
		}

		switch(file_name_idx)
		{
		case 0:
			in_filename = arg;
			break;

		case 1:
			out_filename = arg;
			break;

		default:
			std::cerr << "invalid extra filename" << std::endl;
			return 1;
		}

		++file_name_idx;
	}

	if (min_entries < 1 || max_entries <= min_entries)
	{
		std::cerr << "invalid min/max entries" << std::endl;
		return 3;
	}

	if (entry_size < 1)
	{
		std::cerr << "invalid entry size" << std::endl;
		return 3;
	}

	if (file_name_idx != 2)
	{
		std::cout << "usage: shuffler <infile> <outfile>" << std::endl;
		std::cout << "       --entry n    entry size in bytes, default 38" << std::endl;
		std::cout << "       --min n      minimum number of entries per batch, default 50 million" << std::endl;
		std::cout << "       --max n      maximum number of entries per batch, default 100 million" << std::endl;
		return -1;
	}

	return 0;
}

template<typename T, typename U>
void shuffle_vector(T &rng, std::vector<U> &vec)
{
	if (vec.empty())
		return;

	// fisher-yates shuffle
	for (size_t i=vec.size()-1; i>1; i--)
	{
		auto swidx = rng() % i;
		std::swap(vec[i], vec[swidx]);
	}
}

template<typename T>
void shuffle_buffer(T &rng, std::vector<uint8_t> &buffer, size_t count)
{
	if (!count)
		return;

	std::vector<uint8_t> ebuf;
	ebuf.resize(entry_size);

	// fisher-yates shuffle
	for (size_t i=count-1; i>1; i--)
	{
		auto swidx = rng() % i;

		// swap
		memcpy(ebuf.data(), &buffer[i*entry_size], entry_size);
		memcpy(&buffer[i*entry_size], &buffer[swidx*entry_size], entry_size);
		memcpy(&buffer[swidx*entry_size], ebuf.data(), entry_size);
	}
}

template<typename T>
void create_plan(T &rnd_engine, uint64_t in_size)
{
	plan.clear();

	uint64_t offset = 0;

	while (in_size > 0)
	{
		// note: no uniform distribution but who cares
		auto div = max_entries - min_entries;

		if (div == 0)
			div = 1;

		auto count = rnd_engine() % div + min_entries;
		count *= entry_size;

		if (count > in_size)
			count = in_size;

		plan.push_back({offset, count/entry_size});

		offset += count;
		in_size -= count;
	}

	// shuffle plan now
	shuffle_vector(rnd_engine, plan);

	std::cout << "plan: " << plan.size() << " chunks" << std::endl;
}

int shuffle_main()
{
	std::ifstream f;
	f.open(in_filename.c_str(), std::ios_base::binary | std::ios_base::in);

	if (f.fail())
	{
		std::cerr << "cannot open " << in_filename << std::endl;
		return 4;
	}

	f.seekg(0, std::ios_base::end);
	auto in_size = f.tellg();
	f.seekg(0, std::ios_base::beg);

	std::mt19937_64 rnd_engine;
	rnd_engine.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	create_plan(rnd_engine, in_size);

	std::vector<uint8_t> buffer;
	buffer.resize(max_entries * entry_size);

	std::ofstream fo;
	fo.open(out_filename.c_str(), std::ios_base::binary | std::ios_base::trunc | std::ios_base::out);

	if (fo.fail())
	{
		std::cerr << "cannot create " << out_filename << std::endl;
		return 5;
	}

	size_t step_div = 100;
	size_t step_report = plan.size()/step_div;

	if (step_report == 0)
		step_report = 1;

	size_t next_report = step_report;

	for (size_t i=0; i<plan.size(); i++)
	{
		const auto &it = plan[i];

		if (i == next_report)
		{
			auto percent = (i * 100 + 50) / plan.size();
			std::cout << "plan " << i+1 << " / " << plan.size() << " (" << percent << " %)" << std::endl;
			next_report += step_report;
		}

		f.seekg(it.source_file_offset, std::ios_base::beg);
		f.read((char *)buffer.data(), it.num_entries * entry_size);

		if (f.fail())
		{
			std::cerr << "cannot read " << in_filename << std::endl;
			return 6;
		}

		shuffle_buffer(rnd_engine, buffer, it.num_entries);

		fo.write((char *)buffer.data(), it.num_entries * entry_size);

		if (fo.fail())
		{
			std::cerr << "cannot write " << out_filename << std::endl;
			return 7;
		}
	}

	std::cout << "all done" << std::endl;

	return 0;
}

int main(int argc, const char **argv)
{
	if (int res = parse_args(argc, argv))
		return res;

	return shuffle_main();
}
