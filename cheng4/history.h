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

#pragma once

#include "board.h"

namespace cheng4
{

struct History
{
	static const i16 historyMax = 2047;
	static const i16 historyMin = -historyMax;

	// for counter move queries
	Move previous;

	// history table [stm][piecetype][from][to]
	i16 history[ ctMax ][ ptMax ][ 64 ][ 64 ];
	// counter move table [stm][piecetype][from][to]
	Move counter[ ctMax ][ ptMax ][ 64 ][ 64 ];

	inline History() {}
	explicit inline History( void * /*zeroInit*/ ) { clear(); }

	// add move which caused cutoff/sub move which didn't
	void add( const Board &b, Move m, i32 depth );

	// add counter move for previous m
	void addCounter( const Board &b, Move m, Move cm );

	// get counter move if any
	inline Move getCounter( const Board &b, Move m ) const
	{
		if ( m == mcNull )
			return mcNone;

		Square mf = MovePack::from(m);
		Piece p = b.piece( mf );
		Color c = PiecePack::color( p );
		Piece pt = PiecePack::type(p);

		return counter[c][pt][mf][MovePack::to(m)];
	}

	// get move ordering score
	inline i32 score( const Board &b, Move m ) const
	{
		Square mf = MovePack::from( m );
		Piece p = b.piece( mf );
		return history[ b.turn() ][ PiecePack::type(p) ][ mf ][ MovePack::to(m) ];
	}

	// clear table
	void clear();
};

}
