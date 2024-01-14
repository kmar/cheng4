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

#include "move.h"
#include "magic.h"
#include "zobrist.h"
#include "psq.h"
#include <string>
#include <stdlib.h>
#include <iostream>

// TODO: move large methods to cpp

namespace cheng4
{

class MoveGen;

struct NetCache;

typedef uint UndoMask;

enum UndoFlags
{
	ufIrreversible	=	1,			// restore pawnhash and fifty rule counter
	ufNPMat			=	2,			// restore non-pawn materials
	ufKingState		=	4,			// restore king state (=kingpos and check flag)
	ufCastling		=	8			// restore castling state
};

class Board;
struct Eval;

struct UndoInfo
{
	// holds NetCache
	Eval *eval;
	UndoMask flags;				// undo flags
	// ep square will be restored
	Signature bhash;			// board hash (always)
	Signature phash;			// pawn hash (optional)
	Bitboard bb[7];				// bitboards to restore
	DMat dmat[2];				// original delta-material (always)
	NPMat npmat[2];				// original non-pawn material
	u8 bbi[7];					// indices to restore
	u8 bbCount;					// number of bbs to restore
	Piece pieces[4];			// board pieces to restore
	Square squares[4];			// board squares to restore
	Square ep;					// original ep square (always)
	u8 pieceCount;				// number of board pieces to restore
	FiftyCount fifty;			// original fifty move counter type (if needed)
	CastRights castRights[2];	// original castling rights (if needed)
	u8 kingPos;					// original king position (always)
	bool check;					// original in check flag (always)

	inline UndoInfo()
		: eval(nullptr)
	{
	}

	inline void clear()
	{
		flags = 0;
		bbCount = pieceCount = 0;
	}

	inline void saveBB( u8 index, Bitboard bboard )
	{
		assert( index < bbiMax );
		assert( bbCount < 7 );
		bb[ bbCount ] = bboard;
		bbi[ bbCount++ ] = index;
	}

	inline void savePiece( Square square, Piece piece )
	{
		assert( PiecePack::type( piece ) <= ptKing );
		assert( pieceCount < 4 );
		pieces[ pieceCount ] = piece;
		squares[ pieceCount++ ] = square;
	}
};

class Board
{
protected:
	friend class MoveGen;

	Signature bhash;			// current hash signature
	Signature bpawnHash;		// current pawn hash signature
	Bitboard bb[ bbiMax ];		// bitboard for pieces (indexed using BBI)
	Piece bpieces[ 64 ];		// pieces: color in MSBit
	DMat bdmat[2];				// delta-material [gamephase] - in centipawns
	NPMat bnpmat[2];			// non-pawn material (white, black)
	Square bkingPos[2];			// kings not stored using bitboards
	Color bturn;				// ctWhite(0) or ctBlack(1)
	Square bep;					// enpassant square (new: where opp captures; was a flaw in cheng3)
	FiftyCount bfifty;			// fifty move counter
	CastRights bcastRights[2];	// castling rights for white/black, indexed by colorType
	bool bcheck;				// in check flag

	// extra stuff (not used during search)
	bool frc;					// is fischer random?
	uint curMove;				// current move number

	// castling move is special
	void doCastlingMove( Move move, UndoInfo &ui, bool ischeck );

	inline void initUndo( UndoInfo &ui ) const
	{
		ui.clear();						// clear undo mask
		ui.bhash = bhash;				// hash signature always preserved
		// delta material always preserved
		ui.dmat[ phOpening ] = bdmat[ phOpening ];
		ui.dmat[ phEndgame ] = bdmat[ phEndgame ];
		ui.ep = bep;					// ep square always preserved
	}

	inline void saveKingState( UndoInfo &ui ) const
	{
		assert( !(ui.flags & ufKingState) );
		ui.flags |= ufKingState;
		ui.check = bcheck;
		ui.kingPos = king( turn() );
	}

	template< bool checkFlag > inline void saveCastling( UndoInfo &ui ) const
	{
		if ( checkFlag && ( ui.flags & ufCastling ) )
			return;						// already saved
		assert( !(ui.flags & ufCastling) );
		ui.flags |= ufCastling;
		ui.castRights[ ctWhite ] = castRights( ctWhite );
		ui.castRights[ ctBlack ] = castRights( ctBlack );
	}

	template< bool checkFlag > inline void saveNPMat( UndoInfo &ui ) const
	{
		if ( checkFlag && ( ui.flags & ufNPMat ) )
			return;						// already saved
		assert( !(ui.flags & ufNPMat) );
		ui.flags |= ufNPMat;
		ui.npmat[ ctWhite ] = bnpmat[ ctWhite ];
		ui.npmat[ ctBlack ] = bnpmat[ ctBlack ];
	}

	// also clear fifty rule counter
	inline void saveIrreversible( UndoInfo &ui )
	{
		assert( !(ui.flags & ufIrreversible) );
		ui.phash = bpawnHash;
		ui.fifty = bfifty;
		ui.flags |= ufIrreversible;
		bfifty = 0;
	}

	template< Color c > Draw isDrawByMaterial( MaterialKey mk ) const;

	void calcEvasMask();

public:

	void resetMoveCount()
	{
		bfifty = 0;
		curMove = 1;
	}

	inline uint move() const
	{
		return curMove;
	}

	void setMove(uint val)
	{
		curMove = val;
	}

	// increment move counter by 1
	void incMove();

	// assign castling rights automatically
	void autoCastlingRights();

	inline void setFischerRandom( bool frc_ )
	{
		frc = frc_;
	}

	inline bool fischerRandom() const
	{
		return frc;
	}

	// movegen helper only!
	inline Bitboard evasions() const
	{
		return bb[ bbiEvMask ];
	}

	// simply returns hash signature
	inline Signature sig() const
	{
		return bhash;
	}

	// simply returns pawn hash signature
	inline Signature pawnSig() const
	{
		return bpawnHash;
	}

	// note: doMove methods expect a legal move!

	template< u8 flags > void undoTemplate( const UndoInfo &ui )
	{
		if ( flags & ufIrreversible )
		{
			bpawnHash = ui.phash;
			bfifty = ui.fifty;
		}
		bfifty--;
		if ( flags & ufNPMat )
		{
			bnpmat[ ctWhite ] = ui.npmat[ ctWhite ];
			bnpmat[ ctBlack ] = ui.npmat[ ctBlack ];
		}
		if ( flags & ufKingState )
		{
			bkingPos[ turn() ] = ui.kingPos;
			bcheck = ui.check;
		}
		if ( flags & ufCastling )
		{
			bcastRights[ ctWhite ] = ui.castRights[ ctWhite ];
			bcastRights[ ctBlack ] = ui.castRights[ ctBlack ];
		}
	}

	template< Color color, bool capture, Piece ptype >
		void doMoveTemplate( Move move, Square from, Square to, UndoInfo &ui, bool isCheck );

	// do move
	void doMove( Move move, UndoInfo &ui, bool isCheck );

	// undo move
	void undoMove( const UndoInfo &ui );

	// do null move
	void doNullMove( UndoInfo &ui );

	// undo null move
	void undoNullMove( const UndoInfo &ui );

	// reset to initial position
	void reset();

	// reset to FRC position, indexed from 0 (0..959)
	void resetFRC(int position);

	// clear bits
	void clear();

	// returns 1 if stm is in check
	inline bool inCheck() const
	{
		return bcheck;
	}

	inline MaterialKey materialKey() const
	{
		return (MaterialKey)bb[ bbiMat ];
	}

	// returns delta material for given game phase (fine)
	inline Score deltaMat( Phase ph ) const
	{
		assert( ph <= phEndgame );
		return bdmat[ ph ];
	}

	// return total non-pawn material
	inline NPMat nonPawnMat() const
	{
		return bnpmat[ ctWhite ] + bnpmat[ ctBlack ];
	}

	// non-pawn material for side
	inline NPMat nonPawnMat( Color c ) const
	{
		assert( c <= ctBlack );
		return bnpmat[ c ];
	}

	// return en-passant square
	inline Square epSquare() const
	{
		return bep;
	}

	// return fifty move rule counter
	inline uint fifty() const
	{
		return bfifty;
	}

	void setFifty(uint val)
	{
		bfifty = (FiftyCount)val;
	}

	// reset 50 move rule counter
	void resetFifty()
	{
		bfifty = 0;
	}

	// return side to move
	inline Color turn() const
	{
		return bturn;
	}

	// castling rights
	inline CastRights castRights( Color color ) const
	{
		assert( color <= ctBlack );
		return bcastRights[ color ];
	}

	// returns true if stm can castle
	inline bool canCastle() const
	{
		return castRights( turn() ) != 0;
	}

	// return true if any side can castle
	inline bool canCastleAny() const
	{
		return (castRights(ctWhite) | castRights(ctBlack)) != 0;
	}

	// return occupied mask (all)
	inline Bitboard occupied() const
	{
		return bb[ bbiWOcc ] | bb[ bbiBOcc ];
	}

	// return occupied mask for color
	inline Bitboard occupied( Color c ) const
	{
		assert( c <= ctBlack );
		return bb[ bbiWOcc + c ];
	}

	inline Bitboard pieces( Color color ) const
	{
		assert( color <= ctBlack );
		return bb[ bbiWOcc + color ];
	}

	inline Bitboard pieces( Color color, Piece pt ) const
	{
		assert( pt < ptKing && color <= ctBlack );
		return bb[ BBI( color, pt ) ];
	}

	inline Piece piece( Square sq ) const
	{
		assert( sq < 64 );
		return bpieces[ sq ];
	}

	// clear pieces (used in xboard edit mode)
	void clearPieces();

	// set side to move (used in xboard edit mode)
	void setTurn( Color c );

	// update bitboards from board state (used in xboard edit mode)
	// impl. note: calls update() when done
	void updateBitboards();

	// swap white<=>black
	void swap();

	// set piece of color at square (used in xboard edit mode)
	bool setPiece( Color c, Piece p, Square sq );

	inline Square king( Color color ) const
	{
		assert( color <= ctBlack );
		return bkingPos[ color ];
	}

	inline bool isVacated( Square sq ) const
	{
		return PiecePack::type( piece( sq ) ) == ptNone;
	}

	inline Bitboard diagSliders( Color color ) const
	{
		return pieces( color, ptBishop ) | pieces( color, ptQueen );
	}

	inline Bitboard orthoSliders( Color color ) const
	{
		return pieces( color, ptRook ) | pieces( color, ptQueen );
	}

	inline Bitboard diagSliders() const
	{
		return pieces( ctWhite, ptBishop ) | pieces( ctBlack, ptBishop ) |
			pieces( ctWhite, ptQueen ) | pieces( ctBlack, ptQueen );
	}

	inline Bitboard orthoSliders() const
	{
		return pieces( ctWhite, ptRook ) | pieces( ctBlack, ptRook ) |
			pieces( ctWhite, ptQueen ) | pieces( ctBlack, ptQueen );
	}

	// see helper: get mask for all attacks (both sides) to sq (use occ mask)
	inline Bitboard allAttacksTo( Square sq, Bitboard occ ) const
	{
		Bitboard mask;

		// knights
		mask = Tables::knightAttm[ sq ] & (pieces( ctWhite, ptKnight ) | pieces( ctBlack, ptKnight ));

		// pawns
		mask |= (Tables::pawnAttm[ ctBlack ][ sq ] & pieces( ctWhite, ptPawn )) |
				(Tables::pawnAttm[ ctWhite ][ sq ] & pieces( ctBlack, ptPawn ));

		// sliders
		Bitboard queens = pieces( ctWhite, ptQueen ) | pieces( ctBlack, ptQueen );
		Bitboard orthoSliders = pieces( ctWhite, ptRook ) | pieces( ctBlack, ptRook ) | queens;
		Bitboard diagSliders = pieces( ctWhite, ptBishop ) | pieces( ctBlack, ptBishop ) | queens;

		mask |= diagSliders & Magic::bishopAttm( sq, occ );
		mask |= orthoSliders & Magic::rookAttm( sq, occ );

		// and finally kings
		mask |= Tables::kingAttm[ sq ] & (BitOp::oneShl(king( ctWhite )) | BitOp::oneShl(king( ctBlack )) );

		return mask & occ;
	}

	// returns 1 if square is attacked by color
	template< bool checksliders > bool doesAttack( Color color, Square sq ) const
	{
		Color opc = flip(color);

		// check pawns
		if ( Tables::pawnAttm[ opc ][ sq ] & pieces( color, ptPawn ) )
			return 1;

		// check knights
		if ( Tables::knightAttm[sq] & pieces( color, ptKnight) )
			return 1;

		// check opp king
		if ( Tables::neighbor[ sq ][ king(color) ] )
			return 1;

		if ( checksliders )
		{
			// check sliders
			Bitboard occ = occupied();
			Bitboard sliders;
			sliders = diagSliders( color );
			if ( (Tables::diagAttm[sq] & sliders) &&
				 (Magic::bishopAttm( sq, occ ) & sliders) )
				return 1;
			sliders = orthoSliders( color );
			return
				(Tables::orthoAttm[sq] & sliders) &&
				(Magic::rookAttm( sq, occ ) & sliders);
		} else return 0;
	}

	// returns 1 if square is attacked by color
	template< bool checksliders > bool doesAttack( Color color, Square sq, Bitboard occ ) const
	{
		Color opc = flip(color);

		// check pawns
		if ( Tables::pawnAttm[ opc ][ sq ] & pieces( color, ptPawn ) )
			return 1;

		// check knights
		if ( Tables::knightAttm[sq] & pieces( color, ptKnight) )
			return 1;

		// check opp king
		if ( Tables::neighbor[ sq ][ king(color) ] )
			return 1;

		if ( checksliders )
		{
			// check sliders
			Bitboard sliders;
			sliders = diagSliders( color );
			if ( (Tables::diagAttm[sq] & sliders) &&
				(Magic::bishopAttm( sq, occ ) & sliders) )
				return 1;
			sliders = orthoSliders( color );
			return
				(Tables::orthoAttm[sq] & sliders) &&
				(Magic::rookAttm( sq, occ ) & sliders);
		} else return 0;
	}

	// returns pin mask for pincolor and attackColor
	template< Color pinColor, Color attackColor, bool noqueens > Bitboard pinTemplate( Square sq ) const
	{
		// FIXME: perhaps this could be completely rewritten to make it faster
		// (consider all rays separately)

		Bitboard occ = occupied();

		Bitboard queens = pieces( attackColor, ptQueen );
		Bitboard candidates = pieces( pinColor );
		if ( noqueens )
			candidates &= ~queens;
		Bitboard diagSliders = pieces( attackColor, ptBishop ) | queens;
		Bitboard orthoSliders = pieces( attackColor, ptRook ) | queens;
		Bitboard res = 0;

		if ( diagSliders & Tables::diagAttm[ sq ] )
		{
			Bitboard tmp = Magic::bishopAttm( sq, occ ) & candidates;
			if ( tmp )
			{
				Bitboard tmp2 = Magic::bishopAttm( sq, occ & ~tmp ) & diagSliders;
				while( tmp2 )
					res |= Tables::between[ sq ][ BitOp::popBit( tmp2 ) ] & tmp;
			}
		}

		if ( orthoSliders & Tables::orthoAttm[ sq ] )
		{
			Bitboard tmp = Magic::rookAttm( sq, occ ) & candidates;
			if ( tmp )
			{
				Bitboard tmp2 = Magic::rookAttm( sq, occ & ~tmp ) & orthoSliders;
				while( tmp2 )
					res |= Tables::between[ sq ][ BitOp::popBit( tmp2 ) ] & tmp;
			}
		}

		return res;
	}

	// get checkers from stm's point of view
	inline Bitboard checkers() const
	{
		Bitboard res;

		Color color = turn();
		Square kp = king( color );
		Color opcolor = flip( color );

		// pawns
		res = Tables::pawnAttm[ color ][ kp ] & pieces( opcolor, ptPawn );

		// knights
		res |= Tables::knightAttm[ kp ] & pieces( opcolor, ptKnight );

		// sliders
		Bitboard occ = occupied();

		Bitboard opqueens = pieces( opcolor, ptQueen );
		Bitboard opdiag = pieces( opcolor, ptBishop ) | opqueens;
		Bitboard oportho = pieces( opcolor, ptRook ) | opqueens;

		res |= Magic::bishopAttm( kp, occ ) & opdiag;
		res |= Magic::rookAttm( kp, occ ) & oportho;

		// note: king can't check directly so we're done now

		return res;
	}

	// returns discovered checkers from stm point of view (=stm's dcs)
	inline Bitboard discovered() const
	{
		Color color = turn();
		Square okp = king( flip( color ) );							// opponent's kingpos

		return color == ctWhite ?
			pinTemplate< ctWhite, ctWhite, 1 >( okp )
			: pinTemplate< ctBlack, ctBlack, 1 >( okp );
	}

	// returns pins from stm point of view (=stm's pins)
	inline Bitboard pins() const
	{
		Color color = turn();
		Square kp = king( color );								// my kingpos

		return color == ctWhite ?
			pinTemplate< ctWhite, ctBlack, 0 >( kp )
			: pinTemplate< ctBlack, ctWhite, 0 >( kp );
	}

	// returns 1 if move does check opponent king (from stm point of view)
	bool isCheck( Move m, Bitboard discovered ) const;

	// FIXME: merge later. iisLegal for debugging purposes ATM

	// returns 1 if stm's move is legal
	// used for hashmove/killer validation
	template< bool evasion, bool killer > bool iisLegal( Move m, Bitboard pins ) const;
	template< bool evasion, bool killer > bool isLegal( Move m, Bitboard pins ) const;

	// returns 1 if stm's pseudolegal move is actually legal
	// assume castling moves are always legal => DON'T pass any castling moves!
	template< bool evasion > bool pseudoIsLegal( Move m, Bitboard pins ) const;

	// recompute hashes, material and update board squares from bitboards
	void update();

	// necessary for tuning, fast-update material
	void updateDeltaMaterial();

	// undo psq values from scores (opening, endgames)
	void undoPsq( FineScore *scores ) const;

	// parse from fen
	// returns 0 on error
	const char *fromFEN( const char *fen );

	// returns fen for current position
	std::string toFEN() const;
	// fast version, doesn't add null terminator
	char *toFEN(char *dst) const;

	// move to SAN
	std::string toSAN( Move m ) const;
	// fast version, doesn't add null terminator
	char *toSAN( char *dst, Move m ) const;
	// move to UCI
	std::string toUCI( Move m ) const;
	// fast version, doesn't add null terminator
	char *toUCI( char *dst, Move m ) const;
	// move from UCI
	Move fromUCI( const char *&c ) const;
	Move fromUCI( const std::string &str ) const;
	// move from SAN (includes legality check)
	Move fromSAN( const char *&ptr ) const;
	Move fromSAN( const std::string &str ) const;

	// recompute hash (debug)
	Signature recomputeHash() const;
	// recompute pawn hash (debug)
	Signature recomputePawnHash() const;
	// validate board (debug)
	bool isValid() const;

	// is trivial draw? (material or fifty rule)
	Draw isDraw() const;

	// debug dump
	void dump() const;

	// can do nullmove?
	inline bool canDoNull() const
	{
		// Fonzy's trick
		if ((materialKey() & matNPMask[turn()]) == matKN[turn()])
			return 0;
		return nonPawnMat( turn() ) != 0;
	}

	// can prune move?
	// note: before move is made
	inline bool canPrune( Move m ) const
	{
		// don't prune passer pushes
		Square from = MovePack::from( m );
		Piece p = piece(from);
		if ( PiecePack::type( p ) != ptPawn )
			return 1;
		// don't prune passer pushes
		return (Tables::passerMask[ turn() ][ from ] & pieces( flip(turn()), ptPawn )) != 0;
	}

	// can reduce move?
	// note: move already made
	inline bool canReduce( Move m ) const
	{
		Square to = MovePack::to( m );
		Piece p = piece(to);
		if ( PiecePack::type( p ) != ptPawn )
			return 1;
		// don't reduce passer pushes
		return (Tables::passerMask[ flip(turn()) ][ to ] & pieces( turn(), ptPawn )) != 0;
	}

	// static exchange evaluator
	// fast => used in movegen (where only bad/good capture = sign matters)
	template< bool fast > int see( Move m ) const;

	// returns 1 if move is irreversible
	inline bool isIrreversible( Move m ) const
	{
		return MovePack::isSpecial(m) ||
			PiecePack::type( piece( MovePack::from(m) ) ) == ptPawn;
	}

	bool compare( const Board &tmp ) const;

	// move gain ( used in qs delta(futility) )
	Score moveGain( Move m ) const
	{
		Score res = Tables::gainPromo[ MovePack::promo(m) ];
		return res + Tables::gainCap[ PiecePack::type( piece( MovePack::to(m) ) ) ] * (MovePack::isCapture(m) != 0);
	}

	// get net index for a single piece on a specific square
	// side to move, piece color, piece type, piece square
	i32 netIndex(Color stm, Color c, PieceType pt, Square sq) const;

	// flip net index
	static i32 flipNetIndex(i32 index);

	// get net indices from a specific point of view
	int netIndicesStm(Color stm, i32 *inds) const;

	// note: up to 64 will be set
	// returns number of indices
	// note: indices not sorted
	int netIndices(i32 *inds) const;
	// simple post-validation
	bool validateNetIndices(int ninds, const i32 *inds) const;

	// using netIndex to validate
	int netIndicesDebug(i32 *inds) const;
};

}
