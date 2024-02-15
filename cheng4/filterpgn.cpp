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

#include "filterpgn.h"
#include "board.h"
#include "search.h"
#include "shuffle.h"
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <cctype>

namespace cheng4
{

// FilterPgn::Game

void FilterPgn::Game::clear()
{
	board.reset();
	positions.clear();
	result = -1;
}

// FilterPgn

FilterPgn::FilterPgn()
{
	tt.resize(1);
	s.setHashTable(&tt);
}

bool FilterPgn::write(const char *fname)
{
	std::ofstream ofs( fname, std::ios::out | std::ios::binary );
	if (!ofs.is_open())
		return 0;
	for (size_t i=0; i<globalGame.positions.size(); i++)
	{
		const Position &p = globalGame.positions[i];
		ofs << p.outcome << " " << p.fen << std::endl;
	}
	return 1;
}

bool FilterPgn::parse(const char *fname)
{
	if ( *fname == 32 )
		fname++;
	cheng4::Board b;
	b.reset();
	std::ifstream ifs( fname, std::ios::in | std::ios::binary );
	if (!ifs.is_open())
		return 0;
	ifs.seekg(0, std::ios_base::end);
	std::streampos size = ifs.tellg();
	ifs.seekg(0);
	size_t sz = (size_t)size;
	char *buf = new char[sz+1];
	ifs.read( buf, sz );
	buf[sz] = 0;
	const char *ptr, *top;
	ptr = buf;
	top = ptr + sz;
	ls.ptr = ptr;
	ls.top = top;

	bool forfeit = 0;
	bool parseRes = 1;
	std::string key, value, comment;

	globalGame.clear();
	globalDups.clear();
	game.clear();

	int ch;
	int totalPos = 0;
	int forfeits = 0;
	int illegals = 0;
	// 0 = tags, 1 = game
	int state = 0;
	while ( (ch = peekChar()) >= 0 )
	{
		if (ch == '[') {
			if (state)
			{
				flushGame();
				game.clear();
				state = 0;
			}
			if (!parseTag(key, value))
			{
				parseRes = 0;
				break;
			}

			// parse FEN if provided, necessary for chess960
			if (key == "FEN" && !game.board.fromFEN(value.c_str()))
				fprintf(stderr, "invalid FEN: %s\n", value.c_str());

			// ignore time forfeits and illegal moves
			if (key == "Termination")
			{
				bool isForfeit = value == "time forfeit";
				bool isIllegal = value == "illegal move";

				if (isForfeit || isIllegal)
				{
					forfeit = 1;
					game.result = -1;

					if (isForfeit)
						++forfeits;
					else
						++illegals;
				}
			}

			// we're only interested in result so far
			if (key == "Result" && !forfeit)
			{
				if (value == "1/2-1/2")
					game.result = 0.5;
				else if (value == "1-0")
					game.result = 1;
				else if (value == "0-1")
					game.result = 0;
				else game.result = -1;
			}
			continue;
		}
		if (isspace(ch))
		{
			getChar();
			continue;
		}
		// here we start parsing the game
		state = 1;
		if (ch == '{')
		{
			// parse comment!
			if (!parseComment(comment))
			{
				delete[] buf;
				return 0;
			}
			// here we want to filter book moves and checkmates
			if (comment == "book" && !game.positions.empty())
				game.positions.pop_back();
			if (comment.empty())
				continue;
			const char *c = comment.c_str();
			if ((c[1] == 'M' || c[1] == 'm') && !game.positions.empty())
				game.positions.pop_back();
			continue;
		}
		if (!isalpha(ch))
		{
			// cannot be a move => continue
			getChar();
			continue;
		}
		// so we assume a move goes here
		Move m = game.board.fromSAN(ls.ptr);
		if (m == mcNone)
		{
			// illegal move
			fprintf(stderr, "!");
			getChar();
			continue;
		}
		// we got a legal move here...
		bool ischeck = game.board.isCheck(m, game.board.discovered());
		UndoInfo ui;
		game.board.doMove(m, ui, ischeck);
		if ( game.board.turn() == ctWhite )
			game.board.incMove();

		Position pos;
		pos.fen = game.board.toFEN();
		pos.outcome = game.result;
		game.positions.push_back(pos);

		++totalPos;

		if (!(totalPos & 131071))
			printf("%d positions\n", totalPos);
	}

	printf("%d time forfeits skipped\n", forfeits);
	printf("%d illegal moves skipped\n", illegals);

	flushGame();
	delete[] buf;
	return parseRes;
}

void FilterPgn::flushGame()
{
	if (game.result < 0)
		return;

	std::vector<Position> pos;
	pos.reserve(game.positions.size());

	s.clearHash();
	s.clearSlots();

	for (size_t i=0; i<game.positions.size(); i++)
	{
		const Position &p = game.positions[i];

		s.board.fromFEN(p.fen.c_str());
		s.board.resetMoveCount();
		s.eval.updateNetCache(s.board, s.cacheStack[0].cache);

		Score sc;

		// skip trivial draws
		if (s.board.isDraw())
			continue;

		if (s.board.inCheck())
		{
			// ignore when in check => couldn't call eval in that case
			continue;
		}
		else
			sc = s.qsearch<1, 0>(0, 0, -scInfinity, scInfinity);

		Score sce = s.eval.eval(s.board);

		// ignore non-quiet positions
		if (sce != sc)
			continue;

		// ignore games without outcome
		if (p.outcome < 0)
			continue;

		pos.push_back(p);
	}

	if (pos.empty())
		return;

	static FastRandom rng;
	// shuffle!
	ShuffleArray<Position, FastRandom>(&pos[0], &pos.back()+1, rng);

	for (auto &&it : pos)
	{
		s.board.fromFEN(it.fen.c_str());
		s.board.resetMoveCount();

		// remove dup positions
		if (globalDups.find(s.board.sig()) != globalDups.end())
			continue;

		globalDups.insert(s.board.sig());
		globalGame.positions.push_back(it);
	}
}

bool FilterPgn::parseTag(std::string &key, std::string &value )
{
	int ch = getChar();
	if (ch != '[')
		return 0;

	key.clear();
	value.clear();

	while ( (ch = peekChar()) >= 0 )
	{
		if (isspace(ch)) {
			getChar();
			continue;
		}
		if (ch == ']')
		{
			getChar();
			return 1;
		}
		if (ch == '"')
		{
			if (!parseString(value))
				return 0;
			continue;
		}
		if (!isalpha(ch) && ch != '_')
			return 0;
		// parse key
		do {
			key += (char)getChar();
			ch = peekChar();
		} while (ch >= 0 && (ch == '_' || isalnum(ch)));
	}
	return 0;
}

bool FilterPgn::parseString(std::string &str)
{
	int ch = getChar();
	if (ch != '"')
		return 0;
	str.clear();
	while ( (ch = getChar()) >= 0 )
	{
		if (ch == '"') {
			return 1;
		}
		str += (char)ch;
	}
	// unterminated string
	return 0;
}

bool FilterPgn::parseComment(std::string &str)
{
	int ch = getChar();
	if (ch != '{')
		return 0;
	str.clear();
	while ( (ch = getChar()) >= 0 )
	{
		if (ch == '}') {
			return 1;
		}
		str += (char)ch;
	}
	// unterminated comment
	return 0;
}

}
