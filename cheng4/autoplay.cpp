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

#include "autoplay.h"
#include "movegen.h"
#include "search.h"
#include "thread.h"
#include "shuffle.h"

#include <atomic>
#include <mutex>

namespace cheng4
{

class AutoPlayWorker : public Thread
{
public:
	std::mutex *mutex = nullptr;
	Search *s = nullptr;
	AutoPlay *self = nullptr;
	FILE *file = nullptr;
	int64_t numGames = 0;
	int workerIndex = 0;
	bool doneFlag = 0;

	std::vector<Board> boards;
	std::vector<float> outcomes;
	std::vector<Score> labels;

	FastRandom rng;

	bool genInitialBoard()
	{
		int randplies = 8;

		if ((rng.Next() & 7) == 0)
		{
			// use FRC position
			s->board.resetFRC(int(rng.Next64() % 960));
			randplies = 4;
		}
		else
		{
			s->board.reset();
			randplies = 8;
		}

		for (int i=0; i<randplies; i++)
		{
			if (s->board.isDraw())
				return 0;

			MoveGen mg(s->board);
			Move moves[maxMoves];
			Move m;
			MoveCount count = 0;

			while ((m = mg.next()) != mcNone)
				moves[count++] = m;

			// if checkmated, abort
			if (!count)
				return 0;

			Move move = moves[int(rng.Next64() % count)];

			UndoInfo ui;
			s->board.doMove(move, ui, s->board.isCheck(move, s->board.discovered()));
		}

		if (s->board.isDraw())
			return 0;

		MoveGen mg(s->board);
		if (mg.next() == mcNone)
			return 0;

		return 1;
	}

	void flushBoards()
	{
		mutex->lock();

		self->games += numGames;
		numGames = 0;

		for (size_t i=0; i<boards.size(); i++)
		{
			i16 tmp = (i16)labels[i];
			fwrite(&tmp, 2, 1, file);

			tmp = i16(outcomes[i]*2);
			fwrite(&tmp, 2, 1, file);

			tmp = boards[i].turn() == ctBlack ? 1 : 0;
			fwrite(&tmp, 2, 1, file);

			// compress the board
			uint8_t buf[32];
			boards[i].compressPieces(buf);
			fwrite(buf, 1, 32, file);
		}

		fflush(file);

		self->positions += (int64_t)boards.size();

		double complete = self->positions * 100.0 / self->limit;

		std::cout << self->positions << " positions (" << self->games << " games) " << complete << "% completed" << std::endl;

		if (self->positions >= self->limit)
			doneFlag = 1;

		mutex->unlock();

		boards.clear();
		outcomes.clear();
		labels.clear();
	}

	void playGame()
	{
		Game g;
		s->clearHash();
		s->rep.push(s->board.sig(), true);

		s->board.resetMoveCount();
		g.newGame(&s->board);

		size_t boardStart = boards.size();

		while (!doneFlag)
		{
			if (g.adjudicate())
				break;

			SearchMode sm;
			sm.reset();
			sm.maxNodes = 10000;
			auto tb = s->board;
			Score sc = s->iterate(tb, sm);

			auto move = s->rootMoves.sorted[0]->move;

			Score wsc = sc;

			if (s->board.turn() == ctBlack)
				wsc = -wsc;

			if (!g.doMove(move, wsc))
			{
				assert(0 && "illegal move!");
				break;
			}

			UndoInfo ui;
			s->board.doMove(move, ui, s->board.isCheck(move, s->board.discovered()));

			// avoid overflows
			if (!s->board.fifty())
				s->rep.clear();

			s->rep.push(s->board.sig(), !s->board.fifty());

			// label conditions met?
			if (abs(sc) > 1600 || g.curBoard.inCheck() || MovePack::isSpecial(move) || g.curBoard.move() < 4)
				continue;

			Signature sig = g.curBoard.sig();

			bool dup = false;

			for (auto &it : boards)
			{
				if (it.sig() == sig)
				{
					dup = true;
					break;
				}
			}

			// ignore local dups
			if (dup)
				continue;

			boards.push_back(s->board);
			labels.push_back(sc);
		}

		assert(g.result != -2);

		size_t count = boards.size() - boardStart;

		float outcome = g.result == -1 ? 0.0f : g.result == 0 ? 0.5f : 1.0f;

		for (size_t i=0; i<count; i++)
			outcomes.push_back(outcome);
	}

	void work() override
	{
		auto seed = Timer::getMillisec();
		seed += workerIndex;

		rng.Seed(seed);

		numGames = 0;

		while (!doneFlag)
		{
			s->rep.clear();
			s->board.reset();

			// gen_board_somehow
			while (!genInitialBoard());

			playGame();
			++numGames;

			if (boards.size() >= 16*1024)
				flushBoards();
		}

		flushBoards();
	}
};

void AutoPlay::go(const char *labelFile, int64_t posLimit)
{
	FILE *fo = fopen(labelFile, "wb");

	if (!fo)
		return;

	limit = posLimit;

	const int numThreads = 16;

	std::vector<TransTable *> tt;
	tt.resize(numThreads);

	for (auto &it : tt)
	{
		it = new TransTable;
		// 16 MB
		it->resize( 16*1024*1024 );
	}

	std::vector<Search *> s;
	s.resize(numThreads);

	for (size_t i=0; i<s.size(); i++)
	{
		auto &it = s[i];
		it = new Search;
		it->setHashTable(tt[i]);
	}

	std::vector<AutoPlayWorker *> w;
	w.resize(numThreads);

	std::mutex mutex;

	for (size_t i=0; i<numThreads; i++)
	{
		auto *apw = new AutoPlayWorker;
		apw->workerIndex = (int)i;
		apw->s = s[i];
		apw->mutex = &mutex;
		apw->self = this;
		apw->file = fo;
		w[i] = apw;
	}

	for (auto *it : w)
		it->run();

	for (auto *it : w)
		it->kill();

	for (auto *it : tt)
		delete it;

	for (auto *it : s)
		delete it;

	fclose(fo);
}

}
