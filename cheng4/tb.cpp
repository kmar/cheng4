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

namespace cheng4
{

bool tbInitialized = false;

bool tbInit(const char *path)
{
	if (!path || !*path)
	{
		tbDone();
		return true;
	}

	tbInitialized = true;
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
		tb_free();
	}
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

}
