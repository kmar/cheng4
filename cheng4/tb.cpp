/*
You can use this program under the terms of either the following zlib-compatible license
or as public domain (where applicable)

  Copyright (C) 2012-2015, 2020-2021 Martin Sedlak

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

#include "tb.h"

#include "board.h"

#include "pyrrhic/tbprobe.c"

#include "thread.h"

namespace cheng4
{

bool tbInitialized = false;
Mutex *tbMutex = 0;

bool tbInit(const char *path)
{
	if (!path || !*path)
	{
		tbDone();
		return true;
	}

	tbInitialized = true;

	if (!tbMutex)
		tbMutex = new Mutex;

	return tb_init(path);
}

int tbMaxPieces()
{
	return TB_LARGEST;
}

int tbNumWDL()
{
	return TB_NUM_WDL;
}

void tbDone()
{
	if (tbInitialized)
	{
		tbInitialized = false;
		delete tbMutex;
		tbMutex = 0;
		tb_free();
	}
}

TbProbeResult tbProbeRoot(const Board &board, unsigned *moves)
{
	if (!tbInitialized)
		return tbResInvalid;

	MutexLock lock(*tbMutex);

	// note: not thread-safe, must lock now
	unsigned tres = tb_probe_root(
		BitOp::rowFlip(board.pieces(ctWhite)),
		BitOp::rowFlip(board.pieces(ctBlack)),
		BitOp::rowFlip(BitOp::oneShl(board.king(ctWhite)) | BitOp::oneShl(board.king(ctBlack))),
		BitOp::rowFlip(board.pieces(ctWhite, ptQueen) | board.pieces(ctBlack, ptQueen)),
		BitOp::rowFlip(board.pieces(ctWhite, ptRook) | board.pieces(ctBlack, ptRook)),
		BitOp::rowFlip(board.pieces(ctWhite, ptBishop) | board.pieces(ctBlack, ptBishop)),
		BitOp::rowFlip(board.pieces(ctWhite, ptKnight) | board.pieces(ctBlack, ptKnight)),
		BitOp::rowFlip(board.pieces(ctWhite, ptPawn) | board.pieces(ctBlack, ptPawn)),
		board.fifty(),
		board.epSquare() ? SquarePack::flipV(board.epSquare()) : 0,
		board.turn() == ctWhite,
		moves
	);

	if (tres == TB_RESULT_FAILED || tres == TB_RESULT_CHECKMATE || tres == TB_RESULT_STALEMATE)
		return tbResInvalid;

	return (TbProbeResult)(tres & TB_RESULT_WDL_MASK);
}

TbProbeResult tbProbeWDL(const Board &board)
{
	if (!tbInitialized)
		return tbResInvalid;

	unsigned tres = tb_probe_wdl
	(
		BitOp::rowFlip(board.pieces(ctWhite)),
		BitOp::rowFlip(board.pieces(ctBlack)),
		BitOp::rowFlip(BitOp::oneShl(board.king(ctWhite)) | BitOp::oneShl(board.king(ctBlack))),
		BitOp::rowFlip(board.pieces(ctWhite, ptQueen) | board.pieces(ctBlack, ptQueen)),
		BitOp::rowFlip(board.pieces(ctWhite, ptRook) | board.pieces(ctBlack, ptRook)),
		BitOp::rowFlip(board.pieces(ctWhite, ptBishop) | board.pieces(ctBlack, ptBishop)),
		BitOp::rowFlip(board.pieces(ctWhite, ptKnight) | board.pieces(ctBlack, ptKnight)),
		BitOp::rowFlip(board.pieces(ctWhite, ptPawn) | board.pieces(ctBlack, ptPawn)),
		board.epSquare() ? SquarePack::flipV(board.epSquare()) : 0,
		board.turn() == ctWhite
	);

	if (tres == TB_RESULT_FAILED)
		return tbResInvalid;

	return (TbProbeResult)(tres & TB_RESULT_WDL_MASK);
}

static bool tbConvertSingleMove(const Board &board, unsigned tbmove, Move &move, Score &score)
{
	Square from = SquarePack::flipV((Square)TB_GET_FROM(tbmove));
	Square to = SquarePack::flipV((Square)TB_GET_TO(tbmove));
	Square ep = (Square)TB_GET_EP(tbmove);

	move = mcNone;

	if (ep)
		move = MovePack::initEpCapture(from, to);
	else
	{
		if (board.piece(to) != ptNone)
			move = MovePack::initCapture(from, to);
		else
			move = MovePack::init(from, to);

		unsigned tbpromo = TB_GET_PROMOTES(tbmove);

		Piece promo = ptNone;

		switch(tbpromo)
		{
		case PYRRHIC_PROMOTES_QUEEN:
			promo = ptQueen;
			break;
		case PYRRHIC_PROMOTES_ROOK:
			promo = ptRook;
			break;
		case PYRRHIC_PROMOTES_BISHOP:
			promo = ptBishop;
			break;
		case PYRRHIC_PROMOTES_KNIGHT:
			promo = ptKnight;
			break;
		}

		if (promo != ptNone)
			move |= (Move)promo << msPromo;
	}

	TbProbeResult tbres = (TbProbeResult)TB_GET_WDL(tbmove);

	score = scInvalid;

	switch(tbres)
	{
	case tbResBlessedLoss:
		score = scDraw - 1;
		break;
	case tbResCursedWin:
		score = scDraw + 1;
		break;
	case tbResDraw:
		score = scDraw;
		break;
	case tbResWin:
		score = scTbWin - (Score)TB_GET_DTZ(tbmove);
		break;
	case tbResLoss:
		score = -scTbWin + (Score)TB_GET_DTZ(tbmove);
		break;
	default:
		return false;
	}

	return true;
}

int tbConvertRootMoves(const Board &board, const unsigned *tbmoves, Move *moves, Score *scores)
{
	int res = 0;

	while (tbmoves && *tbmoves != TB_RESULT_FAILED)
	{
		Move move;
		Score score;

		if (!tbConvertSingleMove(board, *tbmoves++, move, score))
			continue;

		if (moves)
			*moves++ = move;

		if (scores)
			*scores++ = score;

		++res;
	}

	return res;
}

}
