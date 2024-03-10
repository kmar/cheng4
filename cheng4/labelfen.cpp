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

#include "labelfen.h"
#include "search.h"
#include "utils.h"
#include "thread.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <atomic>

namespace cheng4
{

bool LabelFEN::load(const char *filename)
{
	if ( *filename == 32 )
		filename++;

	boards.clear();
	outcomes.clear();
	std::cout << "loading " << filename << std::endl;

	std::ifstream ifs( filename, std::ios::in | std::ios::binary );

	if (!ifs.is_open())
		return false;

	ifs.seekg(0, std::ios_base::end);
	std::streampos size = ifs.tellg();
	ifs.seekg(0);
	size_t sz = (size_t)size;
	std::vector<char> buf;
	buf.resize(sz+1);
	ifs.read( buf.data(), sz );

	if (sz < buf.size())
		buf[sz] = 0;

	const char *ptr = buf.data();
	const char *top = &buf[sz];

	Board b;
	b.reset();

	size_t checks = 0;

	while (ptr < top)
	{
		skipSpaces(ptr);
		// "parse" game outcome

		char *end = (char *)ptr;
		double outcome = strtod(ptr, &end);
		ptr = end;

		skipSpaces(ptr);
		ptr = b.fromFEN(ptr);

		if (!ptr)
			return false;

		b.resetFifty();

		skipUntilEOL(ptr);

		if (b.inCheck())
		{
			checks++;
			continue;
		}

		boards.push_back(b);
		outcomes.push_back(outcome);
	}

	std::cout << boards.size() << " positions loaded" << std::endl;
	std::cout << checks << " positions ignored (in check)" << std::endl;

	return true;
}

class LFWorker : public Thread
{
public:
	std::atomic_int *counter = nullptr;
	Search *s = nullptr;
	LabelFEN *lfen;

	void work() override
	{
		auto limit = lfen->boards.size();

		for (;;)
		{
			auto cur = ++*counter - 1;

			if ((size_t)cur >= limit)
				break;

			size_t i = cur;

			if (!(i & 1023))
				std::cout << i << " / " << lfen->boards.size() << std::endl;

			s->rep.clear();
			s->board = lfen->boards[i];

			s->clearHash();
			s->rep.push(s->board.sig(), true);
			SearchMode sm;
			sm.reset();
			sm.maxDepth = 6;
			Score sc = s->iterate(lfen->boards[i], sm);

			lfen->labels[i] = sc;
		}
	}
};

bool LabelFEN::process(const char *outfilename)
{
	labels.resize(boards.size());

	const int numThreads = 16;

	std::vector<TransTable *> tt;
	tt.resize(numThreads);

	for (auto &it : tt)
	{
		it = new TransTable;
		// 1 MB
		it->resize( 1*1024*1024 );
	}

	std::vector<Search *> s;
	s.resize(numThreads);

	for (size_t i=0; i<s.size(); i++)
	{
		auto &it = s[i];
		it = new Search;
		it->setHashTable(tt[i]);
	}

	std::vector<LFWorker *> w;
	w.resize(numThreads);

	std::atomic_int counter(0);

	for (size_t i=0; i<numThreads; i++)
	{
		auto *lfw = new LFWorker;
		lfw->s = s[i];
		lfw->lfen = this;
		lfw->counter = &counter;
		w[i] = lfw;
	}

	for (auto *it : w)
		it->run();

	for (auto *it : w)
		it->kill();

	for (auto *it : tt)
		delete it;

	for (auto *it : s)
		delete it;

	FILE *fo = fopen(outfilename, "wb");

	if (!fo)
		return false;

	size_t outcount = 0;

	for (size_t i=0; i<boards.size(); i++)
	{
		if (abs(labels[i]) > 1600)
			continue;

		i16 tmp = (i16)labels[i];
		fwrite(&tmp, 2, 1, fo);

		i8 tmpi = i8(outcomes[i]*2);
		fwrite(&tmpi, 1, 1, fo);

		tmpi = boards[i].turn() == ctBlack ? 1 : 0;
		fwrite(&tmpi, 1, 1, fo);

		// compress the board
		uint8_t buf[16];
		uint64_t occ = boards[i].compressPiecesOccupancy(buf);

#ifdef _DEBUG
		Board tmpb;
		tmpb.uncompressPiecesOccupancy(occ, buf);
		tmpb.setTurn(Color(tmp ? ctBlack : ctWhite));

		i32 inds0[64];
		auto count0 = boards[i].netIndices(inds0);
		std::sort(inds0, inds0+count0);

		i32 inds1[64];
		auto count1 = tmpb.netIndices(inds1);
		std::sort(inds1, inds1+count1);

		assert(count0 == count1);
		for (int ix=0; ix<count0; ix++)
			assert(inds0[ix] == inds1[ix]);
#endif

		fwrite(&occ, 8, 1, fo);
		fwrite(buf, 1, 16, fo);

		++outcount;
	}

	fclose(fo);

	std::cout << outcount << " total positions output" << std::endl;

	return true;
}


}
