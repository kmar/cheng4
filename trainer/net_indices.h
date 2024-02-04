#pragma once

#include <cstdint>
#include <cassert>

#include "../cheng4/chtypes.h"

// FIXME: trainer version, copied from Board... should clean this up, really

int32_t flipNetIndex(int32_t index)
{
	// ouch, I did something really DUMB with my indices! should've gone for all white, then all black instead of mixed!
	// this will cost me perf
	if (index < 128)
		index ^= 64 ^ 0x38;
	else if (index < 128+48)
	{
		// wpawn
		index -= 128-8;
		index ^= 0x38;
		index += 128-8+48;
	}
	else if (index < 128+2*48)
	{
		// bpawn
		index -= 128-8+48;
		index ^= 0x38;
		index += 128-8;
	}
	else
	{
		index -= 128+2*48;
		index ^= 64 ^ 0x38;
		index += 128+2*48;
	}

	assert(index >= 0 && index < 736);

	return index;
}

int32_t netIndex(cheng4::Color stm, cheng4::Color c, cheng4::PieceType pt, cheng4::Square sq)
{
	assert(pt >= cheng4::ptPawn && pt <= cheng4::ptKing);
	assert(!(sq & ~63));
	assert(!(c & ~1));
	assert(!(stm & ~1));

	c ^= stm;

	cheng4::Square mirror = 0x38 * (stm == cheng4::ctBlack);
	sq ^= mirror;

	switch(pt)
	{
	case cheng4::ptKing:
	{
		// base = 0
		sq += 64*(c == cheng4::ctBlack);
		return sq;
	}

	case cheng4::ptPawn:
	{
		assert(!cheng4::SquarePack::isRank1Or8(sq));
		// base = 128
		return 128 + sq-8 + 48*(c == cheng4::ctBlack);
	}

	case cheng4::ptKnight:
	case cheng4::ptBishop:
	case cheng4::ptRook:
	case cheng4::ptQueen:
	{
		// base
		int base = 128 + 2*48 + (pt - cheng4::ptKnight)*128;
		return base + sq + 64*(c == cheng4::ctBlack);
	}

	default:
		return -1;
	}
}

int netIndicesStm(cheng4::Color stm, const uint8_t buf[32], int16_t *inds)
{
	int res = 0;

	for (int sq=0; sq<64; sq++)
	{
		cheng4::Piece p = cheng4::Piece(buf[sq >> 1]);

		if (sq & 1)
			p >>= 4;

		p &= 15;

		cheng4::Piece pt = cheng4::PiecePack::type(p);

		if (pt == cheng4::ptNone)
			continue;

		inds[res++] = (int16_t)netIndex(stm, cheng4::PiecePack::color(p), cheng4::PieceType(cheng4::PiecePack::type(p)), (cheng4::Square)sq);
	}

	// we don't have to sort for trainer
	//std::sort(inds, inds + res);
	return res;
}

int netIndices(bool btm, const uint8_t buf[32], int16_t *inds)
{
	return netIndicesStm(cheng4::Color(btm ? cheng4::ctBlack : cheng4::ctWhite), buf, inds);
}
