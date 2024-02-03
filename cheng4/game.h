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

#include "board.h"

namespace cheng4
{

struct GameMove
{
	// signature before move was made
	Signature sig;
	Move move;
	// engine score from white's POV (or scInvalid)
	Score score;

	inline GameMove()
		: sig(0)
		, move(mcNone)
		, score(scInvalid)
	{
	}
};

struct Game
{
	// current board
	Board curBoard;
	// startpos board
	Board startBoard;
	// if non-null, adjudication desc
	const char *adjudication;
	// game moves
	std::vector< GameMove > moves;
	// game result, -2 = unknown, -1 = black win, 0 = draw, 1 = white win
	int result;

	// resign score, default = 900
	Score resignScore;
	// number of consecutive moves above resignscore (default = 2)
	int resignMoveCount;
	// minimum move # to test for draw (default = 40)
	int drawMoveNumber;
	// number of consecutive 0.0 moves after drawMoveMin (default = 20)
	int drawMoveCount;

	Game();

	// do move
	bool doMove(Move m, Score sc = scInvalid);

	// new game, optionally from fen
	bool newGame(const Board *board = 0);

	// is threefold repetition?
	bool isThreefold() const;

	// try to adjudicate, returns true if adjudicated
	bool adjudicate();

	// debugging: to pgn
	std::string toPGN() const;

private:
	// adjudication counters
	int resignHalfMoves;
	int drawHalfMoves;
};

}
