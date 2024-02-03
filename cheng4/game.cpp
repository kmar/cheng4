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

#include "game.h"
#include "movegen.h"

namespace cheng4
{

Game::Game()
	: adjudication(0)
	, result(-2)
	, resignScore(900)
	, resignMoveCount(2)
	, drawMoveNumber(40)
	, drawMoveCount(20)
	, resignHalfMoves(0)
	, drawHalfMoves(0)
{
	startBoard.reset();
	curBoard.reset();
}

bool Game::newGame(const Board *board)
{
	adjudication = 0;
	startBoard.reset();

	if (board)
		startBoard = *board;

	resignHalfMoves = drawHalfMoves = 0;

	curBoard = startBoard;

	moves.clear();

	return true;
}

// is threefold repetition?
bool Game::isThreefold() const
{
	Signature sig = curBoard.sig();
	size_t repc = 0;

	for ( i32 i = (i32)moves.size()-2; i >= 0; i-=2 )
	{
		if ( moves[i].sig == sig )
			if ( ++repc >= 2 )
				return 1;
	}

	return 0;
}

bool Game::doMove(Move m, Score sc)
{
	m &= mmNoScore;

	MoveGen mg(curBoard);

	Move gm;

	while ((gm = mg.next()) != mcNone)
	{
		if ( gm == m )
			break;
	}

	if (gm == mcNone)
		return 0;

	GameMove mv;
	mv.sig = curBoard.sig();
	mv.move = m;
	mv.score = sc;

	if (resignScore != scInvalid && abs(sc) >= resignScore)
		++resignHalfMoves;
	else
		resignHalfMoves = 0;

	if (sc == scDraw)
		++drawHalfMoves;
	else
		drawHalfMoves = 0;

	UndoInfo ui;
	bool ischk = curBoard.isCheck(m, curBoard.discovered());
	curBoard.doMove(m, ui, ischk);
	moves.push_back(mv);

	if (curBoard.turn() == ctWhite)
		curBoard.incMove();

	return 1;
}

bool Game::adjudicate()
{
	const Board &b = curBoard;
	Draw dr = b.isDraw();

	if ( dr == drawMaterial )
	{
		adjudication = "result 1/2-1/2 {Insufficient material}";
		result = 0;
		return 1;
	}

	if ( dr == drawFifty )
	{
		adjudication = "result 1/2-1/2 {Fifty move rule}";
		result = 0;
		return 1;
	}

	MoveGen mg(curBoard);
	Move m = mg.next();

	if (m == mcNone)
	{
		if (curBoard.inCheck())
		{
			// mated!
			if ( b.turn() == ctBlack )
			{
				adjudication = "result 1-0 {White mates}";
				result = 1;
			}
			else
			{
				adjudication = "result 0-1 {Black mates}";
				result = -1;
			}
		}
		else
		{
			// stalemate
			adjudication = "result 1/2-1/2 {Stalemate}";
			result = 0;
		}
		return 1;
	}

	// check for draw by repetition!
	if (isThreefold())
	{
		adjudication = "result 1/2-1/2 {Threefold repetition}";
		result = 0;
		return 1;
	}

	// try soft adjudication here
	if (resignMoveCount > 0 && resignHalfMoves >= 2*resignMoveCount)
	{
		if (moves.back().score > 0)
		{
			adjudication = "result 1-0 {White wins by adjudication}";
			result = 1;
		}
		else
		{
			adjudication = "result 0-1 {Black wins by adjudication}";
			result = -1;
		}

		return 1;
	}

	if (drawMoveCount > 0 && drawHalfMoves >= 2*drawMoveCount)
	{
		adjudication = "result 1/2-1/2 {Draw by adjudication}";
		result = 0;
		return 1;
	}

	return 0;
}

std::string Game::toPGN() const
{
	std::string res;
	res += "[Event \"?\"]\n";
	res += "[Result \"";

	switch(result)
	{
	case -1:
		res += "0-1";
		break;
	case 0:
		res += "1/2-1/2";
		break;
	case 1:
		res += "1-0";
		break;
	default:
		res += "*";
	}

	res += "\"]\n";

	//  don't save FEN unless not a standard startpos
	Board start;
	start.reset();

	if (start.sig() != startBoard.sig())
	{
		res += "[FEN \"";
		res += startBoard.toFEN();
		res += "\"]\n";
	}

	Board tb = startBoard;

	// now moves...
	for (size_t i=0; i<moves.size(); i++)
	{
		if (tb.turn() == ctWhite)
		{
			res += std::to_string(tb.move());
			res += ". ";
		}

		Move m = moves[i].move;

		res += tb.toSAN(m);
		res += " ";
		res += "{";
		res += std::to_string(moves[i].score);

		if ((i & 7) == 7)
			res += "}\n";
		else
			res += "} ";

		UndoInfo ui;
		bool ischk = tb.isCheck(m, tb.discovered());
		tb.doMove(m, ui, ischk);

		if (tb.turn() == ctWhite)
			tb.incMove();
	}

	if (adjudication && *adjudication)
	{
		// +7 to skip "result "
		res += adjudication + 7;
	}

	res += "\n";

	return res;
}

}
