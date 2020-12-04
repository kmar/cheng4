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

	// history table [stm][piecetype][square]
	i16 history[ ctMax ][ ptMax ][ 64 ];
	// counter move table [stm][piecetype][square]
	Move counter[ ctMax ][ ptMax ][ 64 ];

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

		Piece p = b.piece( MovePack::from(m) );
		Color c = PiecePack::color( p );
		Piece pt = PiecePack::type(p);

		return counter[c][pt][MovePack::to(m)];
	}

	// get move ordering score
	inline i32 score( const Board &b, Move m ) const
	{
		Piece p = b.piece( MovePack::from( m ) );
		return history[ b.turn() ][ PiecePack::type(p) ][ MovePack::to(m) ];
	}

	// clear table
	void clear();
};

}
