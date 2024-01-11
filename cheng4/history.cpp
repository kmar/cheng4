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

#include "history.h"
#include <memory.h>

namespace cheng4
{

// History

void History::clear()
{
	previous = mcNone;
	memset( history, 0, sizeof(history) );
	memset( counter, 0, sizeof(counter) );
}

void History::addCounter( const Board &b, Move m, Move cm )
{
	Square mf = MovePack::from(m);
	Piece p = b.piece( mf );
	Color c = PiecePack::color( p );
	Piece pt = PiecePack::type(p);

	counter[c][pt][mf][MovePack::to(m)] = cm;
}

void History::add( const Board &b, Move m, i32 depth )
{
	assert( !MovePack::isSpecial(m) );

	Square mf = MovePack::from(m);
	Piece p = b.piece( mf );
	Color c = PiecePack::color( p );
	Piece pt = PiecePack::type(p);

	// might happen if invalid move comes from TT
	if ( pt == ptNone || c != b.turn() )
		return;

	i32 val = depth*depth;			// causing a cutoff higher in the tree is more important
	if ( depth < 0 )
		val = -val;

	i16 &h = history[ c ][ pt ][ mf ][ MovePack::to(m) ];
	i32 nval = (i32)h + val;
	while ( abs(nval) > historyMax )
	{
		nval /= 2;
		for (p=ptPawn; p<=ptKing; p++)
			for ( uint i=0; i<64; i++ )
				for ( uint j=0; j<64; j++ )
					history[c][p][i][j] /= 2;
	}
	h = (i16)nval;
}

}
