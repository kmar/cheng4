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

#include "search.h"
#include "movegen.h"
#include "thread.h"
#include "tune.h"
#include "tb.h"
#include <algorithm>
#include <cassert>
#include <memory.h>
#include <cmath>

namespace cheng4
{

// SearchMode

void SearchMode::reset()
{
	moves.clear();
	mateSearch = 0;
	maxDepth = 0;
	maxTime = 0;
	maxNodes = 0;
	multiPV = 1;
	absLimit = 0;
	ponder = 0;
	fixedTime = 0;
}

bool SearchMode::analyzing() const
{
	return !maxTime && !mateSearch && !maxDepth && !absLimit && !fixedTime;
}

// SearchInfo

void SearchInfo::reset()
{
	flags = 0;
}

// Search::RootMoves

Search::RootMoves &Search::RootMoves::operator =( const RootMoves &o )
{
	discovered = o.discovered;
	count = o.count;
	for ( size_t i=0; i<count; i++ ) {
		moves[i] = o.moves[i];
		sorted[i] = moves + (o.sorted[i] - o.moves);
	}
	bestMove = o.bestMove;
	bestScore = o.bestScore;
	return *this;
}

// Search

enum SearchOpts
{
	useLMR		=	1,
	useNull		=	1,
	useRazoring	=	1,
	useFutility	=	1,
	useSingular	=	1
};

// verbose limit (currently none)
static const i32 verboseLimit = 1000;
// only start sending currmove after this limit
static const i32 currmoveLimit = 1000;

// beta razoring margins
static TUNE_CONST Score betaMargins[] = {
	0, 100, 150, 250, 400, 600, 800
};

// futility margins
static TUNE_CONST Score futMargins[] = {
	0, 100, 150, 250, 400, 600, 800
};

// razoring margins
static TUNE_CONST Score razorMargins[] = {
	0, 150, 200, 250
};

// singular extension margin
static TUNE_CONST Score singularMargin = 26;

// late move futility scale
static TUNE_CONST Score lateMoveFutility = 22;

inline FracDepth Search::lmrFormula(Depth depth, size_t lmrCount)
{
	assert(depth > 0 && lmrCount > 0);

	uint a = BitOp::getMSB(depth);
	uint b = BitOp::getMSB(lmrCount);

	FracDepth res = (FracDepth)(a*b*fracOnePly/3);
	res = (res + fracOnePly/2) & ~(fracOnePly-1);

	if (res > 5*fracOnePly)
		res = 5*fracOnePly;

	return res * (depth*fracOnePly > res);
}

// timed out?
bool Search::timeOut()
{
	++timeOutCounter;
	if ( !(timeOutCounter &= 1023) )
	{
		// check time
		i32 ms = Timer::getMillisec();
		if ( !mode.ponder || ponderHit )
		{
			if ( mode.maxTime )
			{
				i32 delta = ms - startTicks;
				if ( delta >= mode.maxTime )
					return 1;
			}
			if ( mode.maxNodes && nodes >= mode.maxNodes )
				return 1;
		}
		if ( ms - nodeTicks >= 1000 )
		{
			// report nodes now
			i32 dt = ms - startTicks;
			info.reset();
			info.flags |= sifNodes | sifNPS | sifTime | sifHashFull;
			info.nodes = smpNodes();

			info.tbHits = smpTbHits();

			if (info.tbHits)
				info.flags |= sifTB;

			info.nps = dt ? info.nodes * 1000 / dt : 0;
			info.time = (Time)dt;
			info.hashFull = tt->hashFull(age);
			sendInfo();
			nodeTicks = ms;

			// note: moved here so really verboseLimit has to be a multiply of a second
			if ( !verbose && !verboseFixed && dt >= verboseLimit )
			{
				// turning on verbose;
				verbose = 1;
				flushCachedPV( rootMoves.count );
			}
		}
	}
	return 0;
}

template< bool pv, bool incheck > Score Search::qsearch( Ply ply, Depth depth, Score alpha, Score beta )
{
	assert( incheck == board.inCheck() );
	assert( alpha >= -scInfinity && beta <= scInfinity && alpha < beta );

	if ( aborting | abortingSmp )
		return scInvalid;

	uint pvIndex = initPly<pv>( ply );
	if (!pv)
		(void)pvIndex;

	// update selective depth
	if ( ply+1 > selDepth )
		selDepth = ply+1;

	// mate distance pruning
	alpha = std::max( alpha, ScorePack::checkMated(ply) );
	beta = std::min( beta, ScorePack::mateIn(ply) );
	if ( alpha >= beta )
		return alpha;

	// check for timeout
	if ( !(searchFlags & sfNoTimeout) && timeOut() )
	{
		aborting = 1;
		return scInvalid;
	}

	// check for draw first
	if ( isDraw() )
		return scDraw;

	// maximum ply reached?
	if ( ply >= maxPly )
		return scDraw;

	bool qchecks = !incheck && !depth;
#ifndef USE_TUNING
	Depth ttDepth = qchecks ? 0 : -1;

	TransEntry lte;

	Score ttScore = tt->probe( board.sig(), ply, ttDepth, alpha, beta, stack[ply].killers.hashMove, lte );

	if ( !pv && ttScore != scInvalid )
	{
		assert( ScorePack::isValid( ttScore ) );
		return ttScore;
	}
#endif

	Score oalpha;
	if ( pv )
		oalpha = alpha;

	Score positionalBias = 0;

	Score ev;
	Score best;
	if ( incheck )
		best = -scInfinity;
	else
	{
		best = ev =  eval.evalNet( board, alpha, beta );

#ifndef NDEBUG
		// note: this won't work with contempt enabled, of course
		Board tb(board);
		tb.swap();
		// FIXME: ok?
		Score sev = eval.eval( tb, -beta, -alpha );
		if ( sev != ev )
		{
#if 1
			eval.clear();
			ev =  eval.eval( board, alpha, beta );
			sev = eval.eval( tb, -beta, -alpha );
			std::cout << "eval_symmetry_bug!" << std::endl;
			board.dump();
			tb.dump();
#endif
		}
#endif

		assert( !ScorePack::isMate( ev ) );

#ifndef USE_TUNING
		Score ttBetter = TransTable::probeEval( board.sig(), ply, ev, lte );
		if ( ttBetter != scInvalid )
			best = ev = ttBetter;
#endif

		if ( best >= beta )
			return best;			// stand pat
		if ( best > alpha )
			alpha = best;

		positionalBias = abs(eval.fastEval(board) - ev);
	}

	// qsearch limit
	if ( !incheck && depth < minQsDepth )
		return best;

	history->previous = ply > 0 ? stack[ply-1].current : mcNull;
	MoveGen mg( board, stack[ply].killers, *history, qchecks ? mmQCapsChecks : mmQCaps );
	Move m;
	Move bestMove = mcNone;

	size_t count = 0;
	while ( (m = mg.next()) != mcNone )
	{
		stack[ ply ].current = m;
		count++;

		bool ischeck = board.isCheck( m, mg.discovered() );

		// delta/qsearch futility
#ifndef USE_TUNING
		// delta pruning is bad in endgames => require non-pawn material of more than 2 rooks
		if ( useFutility && !pv && !incheck && !ischeck && board.canPrune(m) && board.nonPawnMat() > 10 )
		{
			Score fscore = ev + board.moveGain( m );
			if ( fscore + positionalBias + 100 <= alpha )
				continue;
		}
#endif

		UndoInfo ui;
		eval.netInitUndo(ui, cacheStack[ply+1].cache);
		board.doMove( m, ui, ischeck );
		rep.push( board.sig(), !board.fifty() );

		Score score = ischeck ?
			-qsearch< pv, 1 >( ply+1, depth-1, -beta, -alpha ) :
			-qsearch< pv, 0 >( ply+1, depth-1, -beta, -alpha );

		rep.pop();
		board.undoMove( ui );
		eval.netDoneUndo(cacheStack[ply].cache);

		if ( aborting | abortingSmp )
			return scInvalid;

		if ( score > best )
		{
			best = score;
			if ( score > alpha )
			{
				bestMove = m;
				alpha = score;
				if (pv)
				{
					// copy underlying PV
					triPV[pvIndex] = bestMove;
					copyPV(ply);
				}
				if ( score >= beta )
				{
					// no history here => depth is <= 0
					if ( !MovePack::isSpecial( m ) )
						stack[ply].killers.addKiller( m );
#ifndef USE_TUNING
					tt->store( board.sig(), age, m, score, btLower, ttDepth, ply );
#endif
					return score;
				}
			}
		}
	}

	if ( incheck && !count )
		// checkmated
		return ScorePack::checkMated( ply );

	assert( best > -scInfinity );

#ifndef USE_TUNING
	tt->store( board.sig(), age, bestMove, best, (HashBound)(pv ? (best > oalpha ? btExact : btUpper) : btUpper),
		ttDepth, ply );
#endif

	return best;
}

template< bool pv, bool incheck, bool donull >
	Score Search::search( Ply ply, FracDepth fdepth, Score alpha, Score beta, Move exclude )
{
	assert( incheck == board.inCheck() );
	assert( alpha >= -scInfinity && beta <= scInfinity && alpha < beta );

	if ( aborting | abortingSmp )
		return scInvalid;

	Depth depth = (Depth)(fdepth >> fracShift);

	// qsearch
	if ( depth <= 0 )
		return qsearch< pv, incheck >( ply, 0, alpha, beta );

	uint pvIndex = initPly<pv>( ply );
	if (!pv)
		(void)pvIndex;

	// update selective depth
	if ( ply+1 > selDepth )
		selDepth = ply+1;

	// mate distance pruning
	alpha = std::max( alpha, ScorePack::checkMated(ply) );
	beta = std::min( beta, ScorePack::mateIn(ply) );
	if ( alpha >= beta )
		return alpha;

	// check for timeout
	if ( !(searchFlags & sfNoTimeout) && timeOut() )
	{
		aborting = 1;
		return scInvalid;
	}

	// check for draw first
	if ( isDraw() )
		return scDraw;

	// maximum ply reached?
	if ( ply >= maxPly )
		return scDraw;

	// probe hashtable
	TransEntry lte;
	Score ttScore = tt->probe( board.sig(), ply, depth, alpha, beta, stack[ply].killers.hashMove, lte );
	if ( !pv && !exclude && ttScore != scInvalid )
	{
		assert( ScorePack::isValid( ttScore ) );

		if (ttScore >= beta)
		{
			Move ttmove = stack[ply].killers.hashMove;

			if ( ttmove && board.isLegalMove(ttmove) )
			{
				if (ply > 0 && stack[ply-1].current != mcNull)
					history->addCounter(board, stack[ply-1].current, ttmove);

				if ( !MovePack::isSpecial( ttmove ) )
				{
					stack[ply].killers.addKiller( ttmove );
					history->add( board, ttmove, depth );
				}
			}
		}

		// tt cutoff
		return ttScore;
	}

	// probe tablebases
	if (!(searchFlags & sfNoTablebase) && board.fifty() == 0 && !board.canCastleAny())
	{
		uint numMen = BitOp::popCount(board.occupied());

		if (numMen <= (uint)tbMaxPieces())
		{
			TbProbeResult tbRes = tbProbeWDL(board);

			tbHits += tbRes != tbResInvalid;

			Score tbScore = scInvalid;

			switch(tbRes)
			{
			case tbResBlessedLoss:
				tbScore = scDraw-1;
				break;
			case tbResCursedWin:
				tbScore = scDraw+1;
				break;
			case tbResDraw:
				tbScore = scDraw;
				break;
			case tbResWin:
				tbScore = scTbWin - 100;
				break;
			case tbResLoss:
				tbScore = -scTbWin + 100;
				break;
			default:;
			}

			if (tbScore != scInvalid)
			{
				// store TT, exact
				tt->store( board.sig(), age, mcNone, tbScore, btExact, depth, ply );

				if (!pv || tbRes == tbResDraw)
					return tbScore;
			}
		}
	}

	Score fscore;
	if ( !pv && !incheck )
	{
		fscore = eval.evalNet(board);
		// use more precise tt score if possible
		Score ttBetter = TransTable::probeEval( board.sig(), ply, fscore, lte );
		if ( ttBetter != scInvalid )
			fscore = ttBetter;
		// beta razoring
		Score razEval;
		if ( useRazoring && donull && !exclude && depth <= 6 && (razEval = fscore - betaMargins[depth]) > alpha && !ScorePack::isMate(razEval) )
			return razEval;
	}

	if ( useRazoring && !pv && !incheck && depth <= 3 && stack[ply].killers.hashMove == mcNone &&
		!ScorePack::isMate(alpha) )
	{
		// razoring
		Score margin = razorMargins[depth];

		Score razEval = fscore;

		if ( !exclude && razEval + margin < alpha )
		{
			Score scout = alpha - margin;
			Score score = qsearch< 0, 0 >( ply, 0, scout-1, scout );
			if ( score < scout )
				return score;
		}
	}

	if ( useNull && !pv && !incheck && donull && depth > 1
		&& board.canDoNull() && fscore > alpha && !(searchFlags & sfNoNullMove) )
	{
		// null move pruning
		Depth R = 2 + depth/4;

		UndoInfo ui;
		eval.netInitUndo(ui, cacheStack[ply+1].cache);
		board.doNullMove( ui );
		stack[ ply ].current = mcNull;

		rep.push( board.sig(), 1 );
		Score score = -search< 0, 0, 0 >( ply+1, (depth-R-1) * fracOnePly, -beta, -alpha);
		rep.pop();

		board.undoNullMove( ui );
		eval.netDoneUndo(cacheStack[ply].cache);

		if ( score >= beta )
		{
			if ( depth < 6 )
				return ScorePack::isMate(score) ? beta : score;

			Score vscore = search< 0, 0, 0>( ply, (std::min(depth-5, depth*2/3)) * fracOnePly, alpha, beta);

			if (vscore >= beta)
				return ScorePack::isMate(score) ? beta : score;
		}
	}

	if ((pv || donull) && !exclude && ply > 0 && stack[ply].killers.hashMove == mcNone)
	{
		if (depth > 4)
		{
			// IIR (idea by Ed Schroeder)
			depth--;
			fdepth -= fracOnePly/2;
		}
	}

	Score best = -scInfinity;

	Score oalpha;
	if ( pv )
		oalpha = alpha;

	history->previous = ply > 0 ? stack[ply-1].current : mcNull;
	MoveGen mg( board, stack[ply].killers, *history, mmNormal );
	Move m;
	Move bestMove = mcNone;
	size_t count = 0;			// move count
	size_t lmrCount = 0;
	Move failHist[maxMoves];
	MoveCount failHistCount = 0;

	Move hashmove = stack[ply].killers.hashMove;

	bool doSingular = false;

	if (hashmove)
	{
		Score smargin = singularMargin;
		Score singularAlpha = std::min(alpha, ScorePack::unpackHash(lte.u.s.score, ply));
		singularAlpha -= smargin+1;

		const bool isMateScout = ScorePack::isMate(singularAlpha);

		// isWin limit helps to stabilize fine #70, isMate isn't enough
		// the problem is not scout search but the extension afterwards (TT pressure?)
		// the real problem was my replacement scheme though...
		bool trySingular = useSingular && exclude == mcNone && !isMateScout && depth > 6 && depth+1 < maxDepth &&
			(lte.u.s.bound & 3) >= btLower && !ScorePack::isWin(lte.u.s.score) &&
			lte.u.s.depth > depth/2 &&
			board.isLegal<incheck, false>(hashmove, board.pins());

		doSingular = trySingular &&
			search<false, incheck, false>(ply, fdepth/3, singularAlpha,  singularAlpha+1, hashmove) <= singularAlpha;
	}

	while ( (m = mg.next()) != mcNone )
	{
		stack[ ply ].current = m;
		count++;
		lmrCount = count;

		if (m == exclude)
			continue;

		if ( !MovePack::isSpecial( m ) )
			failHist[ failHistCount++ ] = m;

		bool ischeck = board.isCheck( m, mg.discovered() );

		Score score;

		// extend
		FracDepth extension = std::min( (FracDepth)fracOnePly, extend<pv>( depth, m, ischeck, mg.discovered() ) );

		// singular extension
		if (doSingular && count==1)
			extension = fracOnePly;

		// single evasion extension
		if (incheck && count == 1 && depth > 8 && !mg.peek() && depth < maxDepth-2)
			extension = fracOnePly*3/2;

		FracDepth newDepth = fdepth - fracOnePly + extension;

		if ( useFutility && !pv && !incheck && mg.phase() >= mpQuietBuffer &&
			!extension && depth <= 6 && (!MovePack::isSpecial(m) || MovePack::isUnderPromo(m)) && !ScorePack::isMate(fscore) &&
			board.canPrune(m) )
		{
			// futility pruning
			Score futScore = fscore + futMargins[depth] - (Score)(lateMoveFutility*lmrCount);
			if ( futScore <= alpha )
				continue;

			// SEE pruning
			if (!MovePack::isSpecial(m) && board.see<1>(m) < 0)
				continue;
		}

		i32 hist;
		if ( !incheck )
			hist = history->score(board, m);

		UndoInfo ui;
		eval.netInitUndo(ui, cacheStack[ply+1].cache);
		board.doMove( m, ui, ischeck );
		rep.push( board.sig(), !board.fifty() );

		score = alpha+1;
		if ( pv && count > 1 )
		{
			if ( useLMR && !incheck && mg.phase() >= mpQuietBuffer && (!MovePack::isSpecial(m) || MovePack::isUnderPromo(m)) &&
				!ischeck && depth > 2 && !extension )
			{
				// LMR at pv nodes
				FracDepth reduction = lmrFormula(depth, lmrCount);
				reduction -= fracOnePly * (hist > 0 || !board.canReduce(m) /*|| MovePack::isSpecial(m)*/);

				if (reduction > 0)
					score = -search< 0, 0, 1 >( ply+1, newDepth - reduction, -alpha-1, -alpha );
			}

			if ( score > alpha )
				score = ischeck ?
					-search< 0, 1, 1 >( ply+1, newDepth, -alpha-1, -alpha ) :
					-search< 0, 0, 1 >( ply+1, newDepth, -alpha-1, -alpha );
		}
		// note: reducing bad captures as well
		if ( useLMR && !pv && !incheck && mg.phase() >= mpQuietBuffer &&
			!ischeck && depth > 2 && !extension )
		{
			// LMR at nonpv nodes
			FracDepth reduction = lmrFormula(depth, lmrCount);
			reduction -= fracOnePly * (hist > 0 || !board.canReduce(m) /*|| MovePack::isSpecial(m)*/);

			if (reduction > 0)
				score = -search< 0, 0, 1 >( ply+1, newDepth - reduction, -alpha-1, -alpha );
		}

		if ( score > alpha )
			score = ischeck ?
				-search< pv, 1, !pv >( ply+1, newDepth, -beta, -alpha ) :
				-search< pv, 0, !pv >( ply+1, newDepth, -beta, -alpha );

		rep.pop();
		board.undoMove( ui );
		eval.netDoneUndo(cacheStack[ply].cache);

		if ( aborting | abortingSmp )
			return scInvalid;

		if ( score > best )
		{
			best = score;
			if ( score > alpha )
			{
				bestMove = m;
				alpha = score;
				if (pv)
				{
					// copy underlying PV
					triPV[pvIndex] = bestMove;
					copyPV(ply);
				}
				if ( score >= beta )
				{
					if (exclude)
						return score;

					if (ply > 0 && stack[ply-1].current != mcNull)
						history->addCounter(board, stack[ply-1].current, m);

					if ( !MovePack::isSpecial( m ) )
					{
						stack[ply].killers.addKiller( m );
						history->add( board, m, depth );
						assert( failHistCount > 0 );
						// this useless if is here only to silence msc static analyzer
						if ( failHistCount > 0 )
							failHistCount--;
					}
					for (MoveCount i=0; i<failHistCount; i++)
						history->add( board, failHist[i], -depth);
					tt->store( board.sig(), age, m, score, btLower, depth, ply );
					return score;
				}
			}
		}
	}

	if ( !count )
		// stalemate or checkmate
		return incheck ? ScorePack::checkMated(ply) : scDraw;

	// very important -- we pruned some moves but it's possible that we didn't actually try a move
	// in order to avoid storing wrong mate score to tt, this is necessary
	if ( best == -scInfinity )
	{
		best = alpha;
	}

	assert( best > -scInfinity );

	if (!exclude)
		tt->store( board.sig(), age, bestMove, best,
			(HashBound)(!pv ? btUpper :
			(best > oalpha ? btExact : btUpper)), depth, ply );

	return best;
}

Search::Search( size_t evalKilo, size_t pawnKilo, size_t matKilo ) : startTicks(0), nodeTicks(0),
	timeOutCounter(0), triPV(0), newMultiPV(0), selDepth(0), tt(0), nodes(0), age(0), callback(0),
	callbackParam(0), canStop(0), abortRequest(0), aborting(0), abortingSmp(0),
	outputBest(1), ponderHit(0), maxThreads(511), eloLimit(0), maxElo(2700), contemptFactor(scDraw),
	minQsDepth(-maxDepth), verbose(1), verboseFixed(1), searchFlags(0), startSearch(0), master(0)
{
	cacheStack.resize(maxStack);
	history = new History;
	board.reset();
	mode.reset();
	info.reset();
	for ( int i=0; i<maxMoves; i++ )
		infoPV[i].reset();
	iterBest = iterPonder = mcNone;

	assert( evalKilo * 1024 / 1024 == evalKilo );
	assert( pawnKilo * 1024 / 1024 == pawnKilo );
	assert( matKilo * 1024 / 1024 == matKilo );
	eval.resizeEval( evalKilo * 1024 );
	eval.resizePawn( pawnKilo * 1024 );
	eval.resizeMaterial( matKilo * 1024 );
	eval.clear();

	// init stack
	memset( (void *)stack, 0, sizeof(stack) );
	// allocate triPV
	triPV = new Move[maxTriPV];
	memset( triPV, 0, sizeof(Move)*maxTriPV );

	memset( (void *)&rootMoves, 0, sizeof(rootMoves) );

	history->clear();
}

Search::~Search()
{
	setThreads(0);
	delete[] triPV;
	delete history;
}

void Search::setHashTable(cheng4::TransTable *tt_)
{
	tt = tt_;
}

void Search::clearHash()
{
	assert( tt );
	tt->clear();
}

void Search::clearSlots( bool clearEval )
{
	if ( clearEval )
		eval.clear();
	history->clear();
	memset( (void *)stack, 0, sizeof(stack) );
}

void Search::sendPV( const RootMove &rm, Depth depth, Score score, Score alpha, Score beta, uint mpvindex )
{
	if ( verbose || !verboseFixed )
	{
		i32 dt = Timer::getMillisec() - startTicks;
		SearchInfo &si = verbose ? info : infoPV[mpvindex];

		si.reset();
		// depth required by stupid UCI
		si.flags |= sifDepth | sifSelDepth | sifPV | sifTime | sifNodes | sifNPS;
		si.pvScore = score;
		si.pvBound = (score >= beta) ? btLower : (score <= alpha) ? btUpper : btExact;
		si.pvIndex = mpvindex;
		si.pvCount = rm.pvCount;
		si.pv = rm.pv;
		si.nodes = smpNodes();

		si.tbHits = smpTbHits();

		if (si.tbHits)
			si.flags |= sifTB;

		si.nps = dt ? si.nodes * 1000 / dt : 0;
		si.time = (Time)dt;
		Ply sd = selDepth;
		for (size_t i=0; i<smpThreads.size(); i++)
			sd = std::max( sd, smpThreads[i]->search.selDepth);
		// note: this feels a bit hacky, but we adjust main search seldepth here to be max of smp threads as well
		selDepth = std::max<Ply>(selDepth, sd);
		si.depth = depth;
		si.selDepth = sd;
		if ( verbose )
			sendInfo();
	}
}

void Search::flushCachedPV( size_t totalMoves )
{
	for ( size_t i=0; i<totalMoves; i++ ) {
		SearchInfo &si = infoPV[i];
		if ( si.flags ) {
			sendInfo( si );
			si.reset();
		}
	}
}

static void searchOverrideTBScore(Move rm, Score &score, int tbMoveCount, const Move *tbMoves, const Score *tbScores)
{
	// if we found a mate, just use that
	if (score != -scInfinity && ScorePack::isMate(score))
		return;

	for (int i=0; i<tbMoveCount; i++)
	{
		if (tbMoves[i] == rm)
		{
			if (tbScores[i] != scInvalid)
				score = tbScores[i];

			break;
		}
	}
}

// do root search
Score Search::root( Depth depth, Score alpha, Score beta )
{
	// ply = 0 here

	// qsearch explosion guard
	minQsDepth = (Depth)-std::min( (int)maxDepth, (int)(depth*3));

	rootMoves.bestMove = mcNone;
	rootMoves.bestScore = scInvalid;

	initPly<1>( 0 );

	Score oalpha = alpha;

	FracDepth fd = (FracDepth)depth << fracShift;

	Score best = scInvalid;
	Move bestm = mcNone;

	// tb lookup now
	Move tbMoves[maxMoves];
	Score tbScores[maxMoves];
	int tbMoveCount = 0;

	// tablebase root probe and score
	if (!(searchFlags & sfNoTablebase) && !board.canCastleAny())
	{
		uint numMen = BitOp::popCount(board.occupied());

		if (numMen <= (uint)tbMaxPieces())
		{
			unsigned ltbMoves[maxMoves];
			TbProbeResult tbres = tbProbeRoot(board, ltbMoves);

			if (tbres != tbResInvalid)
				tbMoveCount = tbConvertRootMoves(board, ltbMoves, tbMoves, tbScores);

			// only override all for depth 1
			for (size_t i=0; depth == 1 && i<rootMoves.count; i++)
				searchOverrideTBScore(rootMoves.sorted[i]->move, rootMoves.sorted[i]->score, tbMoveCount, tbMoves, tbScores);
		}
	}

	if (tbMoveCount > 0)
	{
		// disable aspiration for root tbhits
		oalpha = alpha = -scInfinity;
		beta = scInfinity;
	}

	// first thing to do: sort root moves
	std::stable_sort( rootMoves.sorted, rootMoves.sorted + rootMoves.count, rootPred );
	for (size_t i=0; i<rootMoves.count; i++)
	{
		RootMove &rm = *rootMoves.sorted[i];
		if ( i >= mode.multiPV )
			rm.score = -scInfinity;
	}

	size_t count = 0;

	Ply maxSelDepth = 0;

	for (size_t i=0; i<rootMoves.count; i++)
	{
		count++;
		RootMove &rm = *rootMoves.sorted[i];

		stack[ 0 ].current = rm.move;

		if ( verbose )
		{
			i32 dt = Timer::getMillisec() - startTicks;
			if ( dt >= currmoveLimit )
			{
				info.reset();
				info.flags |= sifCurIndex | sifCurMove;
				info.curIndex = (MoveCount)i;
				info.curCount = (MoveCount)rootMoves.count;
				info.curMove = rm.move;
				sendInfo();
			}
		}

		NodeCount onodes = nodes;

		Score score;

		bool isCheck = board.isCheck( rm.move, rootMoves.discovered );

		// extend
		FracDepth extension = extend<1>( depth, rm.move, isCheck, rootMoves.discovered );
		FracDepth newDepth = fd - fracOnePly + extension;

		UndoInfo ui;
		eval.netInitUndo(ui, cacheStack[1].cache);

		board.doMove( rm.move, ui, isCheck );
		rep.push( board.sig(), !board.fifty() );

		score = alpha+1;
		if ( count > mode.multiPV )
		{
			score = isCheck ?
				-search< 0, 1, 1 >( 1, newDepth, -alpha-1, -alpha ) :
				-search< 0, 0, 1 >( 1, newDepth, -alpha-1, -alpha );
		}
		if ( score > alpha )
		{
			score = isCheck ?
				-search< 1, 1, 0 >( 1, newDepth, -beta, -alpha ) :
				-search< 1, 0, 0 >( 1, newDepth, -beta, -alpha );
		}
		rep.pop();
		board.undoMove( ui );
		eval.netDoneUndo(cacheStack[0].cache);

		selDepth = maxSelDepth = std::max<Ply>(maxSelDepth, selDepth);

		if ( aborting )
			return scInvalid;
		if ( abortingSmp )
			break;

		if (tbMoveCount)
			searchOverrideTBScore(rm.move, score, tbMoveCount, tbMoves, tbScores);

		rm.nodes = nodes - onodes;

		if ( count == 1 && score <= alpha )
		{
			// FIXME: break here?!
			rootMoves.bestMove = triPV[0] = rm.move;
			triPV[1] = mcNone;
			// extract pv now
			extractPV( rm );
			sendPV( rm, depth, score, oalpha, beta );
			if ( master )
			{
				master->abortingSmp = 1;
				rm.score = score;
			}
			return score;		// early exit => fail low!
		}

		if ( score > best )
		{
			bestm = rm.move;
			best = score;
		}
		if ( score > alpha )
		{
			// copy underlying PV
			triPV[0] = rm.move;
			copyPV(0);

			alpha = score;
			rm.score = score;

			// extract pv now
			extractPV( rm );

			if ( mode.multiPV <= 1 )
			{
				sendPV( rm, depth, score, oalpha, beta );

				// make sure rm is first now!
				for (size_t j=i; j>0; j--)
					rootMoves.sorted[j] = rootMoves.sorted[j-1];
				rootMoves.sorted[0] = &rm;
			}
			else
			{
				// special (multipv mode)
				// first, determine if this move makes it into new multipv
				// if not, don't do anything
				uint mpv = std::min( (uint)count, mode.multiPV );
				uint pvcount = 0;
				bool ok = 0;
				for ( size_t j=0; j<mpv; j++ )
				{
					Score mscore = rootMoves.sorted[j]->score;
					if ( mscore != -scInfinity )
						pvcount++;
					if ( score >= mscore )
						ok = 1;
				}
				mpv = std::min( mpv, pvcount );
				if ( ok )
				{
					// yes, we're updating multipv move score
					std::stable_sort( rootMoves.sorted, rootMoves.sorted + count, rootPred );
					// FIXME: if in xboard mode, could send the PV right away
					// doing the stupid UCI stuff shouldn't hurt probably
					if ( mpv >= mode.multiPV )
					{
						// send PVs
						for (size_t j=0; j<mpv; j++)
						{
							sendPV(*rootMoves.sorted[j], depth, rootMoves.sorted[j]->score,
								-scInfinity, scInfinity, (uint)j);
						}
					}
					// make sure alpha is up to date
					if ( mpv < mode.multiPV )
						alpha = -scInfinity;
					else
					{
						// we're really only interested in moves that beat worst multiPV move
						Score newAlpha = scInfinity;
						for (size_t j=0; j<mpv; j++)
							newAlpha = std::min( newAlpha, rootMoves.sorted[j]->score );
						alpha = newAlpha;
					}
				}
			}

			// set -inf score to uninteresting moves
			for (size_t j=mode.multiPV; j<rootMoves.count; j++)
					rootMoves.sorted[j]->score = -scInfinity;

			if ( score >= beta )
			{
				// cutoff
				rootMoves.bestMove = bestm;
				rootMoves.bestScore = best;

				// don't store if we're searching a subset of root moves!
				if ( mode.moves.empty() )
					tt->store( board.sig(), age, bestm, best, btLower, depth, 0 );

				if ( master )
					master->abortingSmp = 1;

				return best;
			}
		}
	}

	if ( abortingSmp )
	{
		for (size_t i=0; i<smpThreads.size(); i++)
		{
			const RootMoves &rm = smpThreads[i]->search.rootMoves;
			if ( rm.bestMove == mcNone )
				continue;
			best = rm.bestScore;
			rootMoves = rm;

			// we have to reset cached pvs if any
			for (size_t j=0; j<rootMoves.count; j++)
				infoPV[j].reset();

			if ( rootMoves.count )
				for (uint j=0; j<mode.multiPV; j++)
					sendPV( *rootMoves.sorted[j], depth, rootMoves.sorted[j]->score, oalpha, beta, j );

			// FIXME: break here?
			return best;
		}
		assert( bestm != mcNone );
		if ( bestm == mcNone )
			return scInvalid;				// paranoid
	}

	rootMoves.bestMove = bestm;
	rootMoves.bestScore = best;

	// don't store if we're searching a subset of root moves!
	if ( rootMoves.count && mode.moves.empty() )
		tt->store( board.sig(), age, bestm, best, (HashBound)(best > oalpha ? btExact : btUpper), depth, 0 );

	if ( master )
		master->abortingSmp = 1;

	return best;
}

i32 Search::initIteration()
{
	i32 sticks = Timer::getMillisec();

	iterBest = iterPonder = mcNone;
	canStop = 0;
	abortRequest = 0;
	aborting = 0;
	outputBest = 1;
	ponderHit = 0;
	verbose = 0;
	verboseFixed = 1;

	timeOutCounter = 1023;
	nodeTicks = startTicks = sticks;

	nodes = 0;
	tbHits = 0;

	// increment age
	age++;
	tt->clearHashFull();

	return sticks;
}

static History rootHist(0);

Score Search::iterate( const Board &b, const SearchMode &sm, bool nosendbest )
{
	assert( tt );

	Score res = scInvalid;

	Score contempt = scDraw;

	// ignore contempt when analyzing (=infinite time)
	if ( !sm.analyzing() )
		contempt = b.turn() == ctWhite ? contemptFactor : -contemptFactor;

	eval.setContempt( contempt );

	selDepth = 0;
	for (size_t i=0; i<smpThreads.size(); i++)
	{
		smpThreads[i]->search.selDepth = 0;
		smpThreads[i]->search.eval.setContempt( contempt );
	}

	initIteration();
	startSearch.signal();

	// don't output anything if we should think for a limited amount of time
	verbose = verboseFixed = !sm.maxTime || eloLimit;

	board = b;
	eval.updateNetCache(board, cacheStack[0].cache);
	mode = sm;

	Killer killers(0);

	rootMoves.moves[0].move = mcNone;
	rootMoves.count = 0;

	// init with hashmove if available
	tt->probe( b.sig(), 0, 0, scDraw, scDraw, killers.hashMove );

	// init root moves
	MoveGen mg( b, killers, rootHist );
	Move m;
	rootMoves.discovered = mg.discovered();

	while ( (m = mg.next()) != mcNone )
	{
		if ( !sm.moves.empty() && std::find(sm.moves.begin(), sm.moves.end(), m) == sm.moves.end() )
			continue;
		if ( !verboseFixed )
			infoPV[ rootMoves.count ].reset();
		RootMove &rm = rootMoves.moves[ rootMoves.count++ ];
		rm.nodes = 0;
		rm.score = -scInfinity;
		rm.move = m;
		rm.pv[0] = mcNone;
		rm.pvCount = 0;
	}

	for (size_t i=0; i<rootMoves.count; i++)
		rootMoves.sorted[i] = rootMoves.moves + i;

	Depth depthLimit = maxDepth;

	if ( mode.maxTime && rootMoves.count == 1 && sm.moves.empty() ) {
		// play only move fast. using depth 2 to have (at least) something to ponder on
		depthLimit = 2;
	}

	if ( mode.maxDepth )
		depthLimit = std::min( depthLimit, mode.maxDepth );

	// limit multiPV to number of available moves!
	mode.multiPV = std::min( mode.multiPV, (uint)rootMoves.count );

	smpSync();

	Score lastIteration = scDraw;		// last iteration score

	i32 lastIterationStart = startTicks;

	for ( Depth d = 1; rootMoves.count && d <= depthLimit; d++ )
	{
		for (size_t i=0; i<smpThreads.size(); i++)
			smpThreads[i]->search.selDepth = 0;

		i32 curTicks = Timer::getMillisec();
		i32 lastIterationDelta = curTicks - lastIterationStart;
		lastIterationStart = curTicks;

		// update multiPV on the fly if needed
		if ( newMultiPV )
		{
			// limit multiPV to number of available moves!
			mode.multiPV = newMultiPV;
			mode.multiPV = std::min( mode.multiPV, (uint)rootMoves.count );
			newMultiPV = 0;
		}

		i32 total = curTicks - startTicks;

		if ( d > 1 && (!mode.ponder || ponderHit) && mode.maxTime && !mode.fixedTime )
		{
			// make sure we can at least finish first move on this iteration,
			// assuming it will take 50% of current iteration (=100% previous iteration)
			if ( total + lastIterationDelta > mode.maxTime )
				break;
		}

		i32 limitStart = 0;
		if ( eloLimit && maxElo < (u32)maxStrength )
			limitStart = curTicks;

		if ( verbose )
		{
			info.reset();
			info.flags |= sifDepth | sifSelDepth | sifTime;
			info.depth = d;
			info.selDepth = selDepth;
			info.time = (Time)total;
			sendInfo();
		}

		// reset selDepth
		selDepth = 0;

		if ( d == 1 )
		{
			// depth 1: always full
			// disable timeout here
			abortingSmp = 0;
			enableTimeOut(0);
			res = lastIteration = root( d, -scInfinity, +scInfinity );
			enableTimeOut(1);
			canStop = 1;
			if ( abortRequest )
				aborting = 1;
		}
		else if ( mode.multiPV > 1 )
		{
			// lazySMP kicks in here
			smpStart( d, -scInfinity, scInfinity );
			res = root( d, -scInfinity, scInfinity );
			smpStop();
		}
		else
		{
			// aspiration search
			Score prevScore = lastIteration;

			Score window = 15;
			Score alpha = lastIteration - window;
			Score beta = lastIteration + window;

			i32 maxTime = mode.maxTime;
			bool blunderCheck = false;
			bool failHigh = false;

			Ply maxSelDepth = 0;

			for (;;)
			{
				alpha = std::max( -scInfinity, alpha );
				beta = std::min( +scInfinity, beta );
				assert( alpha < beta );

				// lazySMP kicks in here
				smpStart( d, alpha, beta );
				Score score = root( d, alpha, beta );
				selDepth = maxSelDepth = std::max<Ply>(maxSelDepth, selDepth);
				smpStop();

				if ( aborting )
					break;

				res = score;

				if ( score > alpha && score < beta )
				{
					lastIteration = score;
					break;
				}

				window *= 2;

				if ( score <= alpha )
				{
					// fail low
					while (alpha - window < -scInfinity)
						window /= 2;

					alpha -= window;
					beta -= window/3;

					// fail low after fail high triggers blunder check - saw Cheng lose 1 game because of this
					if ( failHigh || abs( score - prevScore ) >= 30 )
					{
						// blunder warning => give more time to resolve iteration (if possible)
						blunderCheck = 1;
						mode.maxTime = mode.absLimit;
					}
				}
				else
				{
					failHigh = true;
					// fail high
					while (beta + window > scInfinity)
						window /= 2;

					beta += window;
					alpha += window/3;
				}
			}
			if ( blunderCheck )
				mode.maxTime = maxTime;
		}
		if ( eloLimit && maxElo < (u32)maxStrength )
		{
			// we're in elo limit mode elo => add artificial slowdown!
			i32 ticks = Timer::getMillisec();
			i32 delta = ticks - limitStart;
			i32 slowdown = (i32)maxStrength - (i32)maxElo;
			// -100 = 2x slower
			// -200 = 4x slower
			// -300 = 8x slower
			// -400 = 16x slower
			// -500 = 32x slower
			// -600 = 64x slower
			// -700 = 128x slower
			// -800 = 256x slower
			// -900 = 512x slower
			// -1000 = 1024x slower
			// -1100 = 2048x slower
			// -1200 = 4096x slower
			// -1300 = 8192x slower
			// -1400 = 16384x slower
			// -1500 = 32768x slower
			// -1600 = 65536x slower
			// -1700 = 131072x slower
			// .. etc
			i64 delay = (i64)(floor(pow(2.0, slowdown/100.0) * std::max(delta, 1)) + 0.5);
			delay -= delta + 1;			// adjust for sleep granularity
			if ( sm.maxTime && ticks + delay - startTicks >= sm.maxTime )
				aborting = 1;			// early exit => we won't be able to reach next iteration anyway
			while ( delay > 0 && !aborting && eloLimit && maxElo < (u32)maxStrength  )
			{
				Thread::sleep(1);
				// hack to force timeout check
				timeOutCounter = 1023;
				if ( timeOut() )
				{
					aborting = 1;
					break;
				}
				i32 current = Timer::getMillisec();
				delay -= current - ticks;
				ticks = current;
			}
		}

		if ( mode.mateSearch )
		{
			Score score =  mode.multiPV ? res : lastIteration;

			int msc = score >= 0 ? (scInfinity - score)/2 + 1 : (-scInfinity - score + 1)/2 - 1;

			if ( msc == (int)mode.mateSearch )
				break;
		}

		if ( aborting )
			break;
	}

	flushCachedPV( rootMoves.count );

	if ( mode.ponder )
	{
		// wait for stop or ponderhit!
		// FIXME: better! (use event)
		while ( !aborting && !ponderHit )
			Thread::sleep(1);
	}

	// finally report total nodes and time
	if ( verbose )
	{
		i32 dt = Timer::getMillisec() - startTicks;
		info.reset();
		info.flags |= sifTime | sifNodes | sifNPS;
		info.nodes = smpNodes();

		info.tbHits = smpTbHits();

		if (info.tbHits)
			info.flags |= sifTB;

		info.nps = dt ? info.nodes * 1000 / dt : 0;
		info.time = (Time)dt;
		sendInfo();
	}

	if ( outputBest )
	{
		// return best move and ponder move (if available)
		if ( !rootMoves.sorted[0] )
		{
			iterBest = iterPonder = mcNone;
			if ( !nosendbest )
				sendBest();
			return res;
		}
		const RootMove &rm = *rootMoves.sorted[0];
		iterBest = rm.move;
		if ( rm.pvCount > 1 )
			iterPonder = rm.pv[1];
		if ( iterPonder == mcNone && iterBest != mcNone && rootMoves.count )
		{
			// we don't have a move to ponder on!
			// this can happen if we're in the middle of resolving a fail-low or fail-high and time is up
			// so try to extract it from hashtable!
			iterPonder = extractPonderFromHash( iterBest );
		}
		if ( !nosendbest )
			sendBest();
	}
	return res;
}

void Search::getBest( SearchInfo  &sinfo )
{
	sinfo.reset();
	sinfo.flags |= sifBestMove;
	sinfo.bestMove = iterBest;
	if ( iterPonder != mcNone )
	{
		sinfo.flags |= sifPonderMove;
		sinfo.ponderMove = iterPonder;
	}
	iterBest = iterPonder = mcNone;
}

void Search::sendBest()
{
	getBest( info );
	sendInfo();
}

void Search::setCallback( SearchCallback cbk, void *param )
{
	callback = cbk;
	callbackParam = param;
}

// execute info callback
void Search::sendInfo( const SearchInfo &sinfo ) const
{
	if ( !sinfo.flags || !callback )
		return;
	callback( sinfo, callbackParam );
}

void Search::sendInfo() const
{
	sendInfo( info );
}

// copy underlying PV
void Search::copyPV( Ply ply )
{
	uint index = triIndex(ply);
	Move *dst = triPV + index + 1;
	const Move *src = triPV + index + (maxPV-ply);
	assert( index + (maxPV-ply) == triIndex(ply+1) );
	do
	{
		*dst++ = *src;
	} while ( *src++ );

#if 0
	// validate PV
	auto tmp = board;
	auto *srcpv = triPV + index;

	while (*srcpv)
	{
		auto mv = *srcpv++;

		if (!tmp.isLegalMove(mv))
		{
			printf("illegal move in PV!!!\n");
			tmp.dump();
			printf("move: %s\n", tmp.toSAN(mv).c_str());
			break;
		}

		UndoInfo ui;
		tmp.doMove(mv, ui, tmp.isCheck(mv, tmp.discovered()));
	}
#endif
}

// extract PV for rm
void Search::extractPV( RootMove &rm ) const
{
	rm.pvCount = 0;
	const Move *src = triPV;
	assert( *src );

	Board tb = board;

	while ( *src )
	{
		// FIXME: this should never happen, but it apparently does happen in cutechess-cli sometimes
		if (!tb.isLegalMove(*src))
		{
			assert(0&&"got illegal move in PV!");
			break;
		}

		UndoInfo ui;
		tb.doMove(*src, ui, tb.isCheck(*src, tb.discovered()));

		rm.pv[ rm.pvCount++ ] = *src++;
	}

	rm.pv[ rm.pvCount ] = mcNone;
}

Move Search::extractPonderFromHash( Move best )
{
	Move ponder;
	UndoInfo ui;
	board.doMove( best, ui, board.isCheck( best, board.discovered()) );
	tt->probe(board.sig(), 0, 0, scDraw, scDraw, ponder);
	if ( ponder != mcNone )
	{
		// found - now make sure it's legal
		bool legal = board.isLegalMove(ponder);

		if ( !legal )
			ponder = mcNone;
	}
	board.undoMove(ui);
	return ponder;
}

// set helper threads
void Search::setThreads( size_t nt )
{
	if ( nt > maxThreads )
		nt = maxThreads;
	if ( smpThreads.size() == nt )
		return;
	for ( size_t i=0; i < smpThreads.size(); i++)
		smpThreads[i]->kill();
	smpThreads.clear();
	for ( size_t i=0; i < nt; i++)
	{
		LazySMPThread *smpt = new LazySMPThread;
		smpt->search.master = this;
		smpt->search.setHashTable( tt );
		smpt->run();
		smpThreads.push_back( smpt );
	}
}

// set maximum # of helper threads
void Search::setMaxThreads( size_t maxt )
{
	maxThreads = maxt;
	setThreads( smpThreads.size() );
}

// set multipv (can also be called while analyzing!)
void Search::setMultiPV( uint mpv )
{
	newMultiPV = mpv;
}

void Search::setEloLimit( bool limit )
{
	eloLimit = limit;
}

void Search::setMaxElo( u32 elo )
{
	maxElo = elo;
}

void Search::setContempt( Score contempt )
{
	contemptFactor = contempt;
}

// enable timeout
void Search::enableTimeOut( bool enable )
{
	if ( !enable )
		searchFlags |= sfNoTimeout;
	else
		searchFlags &= ~sfNoTimeout;
}

void Search::disableTablebase( bool flag )
{
	if ( flag )
		searchFlags |= sfNoTablebase;
	else
		searchFlags &= ~sfNoTablebase;
}

// enable nullmove
void Search::enableNullMove( bool enable )
{
	if ( !enable )
		searchFlags |= sfNoNullMove;
	else
		searchFlags &= ~sfNoNullMove;
}

void Search::smpStart( Depth depth, Score alpha, Score beta )
{
	abortingSmp = 0;
	for ( size_t i=0; i<smpThreads.size(); i++ )
		smpThreads[i]->start( depth + (Depth)((i&1)^1), alpha, beta, *this );
}

void Search::smpStop()
{
	for ( size_t i=0; i<smpThreads.size(); i++ )
		smpThreads[i]->abort();
}

void Search::smpSync() const
{
	for ( size_t i=0; i<smpThreads.size(); i++)
	{
		// synchronize smp threads
		assert( !smpThreads[i]->searching );
		Search &s = smpThreads[i]->search;
		s.initIteration();
		s.age = age;
		s.board = board;
		s.eval.updateNetCache(board, s.cacheStack[0].cache);
		// FIXME: better?
		s.rep.copyFrom(rep);
		*s.history = *history;
		s.rootMoves = rootMoves;
		// never use timeout for smp helper threads!
		s.searchFlags = searchFlags | sfNoTimeout;
	}
}

// static init
void Search::init()
{
	TUNE_EXPORT(Score, razorMargin1, razorMargins[1]);
	TUNE_EXPORT(Score, razorMargin2, razorMargins[2]);
	TUNE_EXPORT(Score, razorMargin3, razorMargins[3]);

	TUNE_EXPORT(Score, futMargin1, futMargins[1]);
	TUNE_EXPORT(Score, futMargin2, futMargins[2]);
	TUNE_EXPORT(Score, futMargin3, futMargins[3]);
	TUNE_EXPORT(Score, futMargin4, futMargins[4]);
	TUNE_EXPORT(Score, futMargin5, futMargins[5]);
	TUNE_EXPORT(Score, futMargin6, futMargins[6]);

	TUNE_EXPORT(Score, betaMargin1, betaMargins[1]);
	TUNE_EXPORT(Score, betaMargin2, betaMargins[2]);
	TUNE_EXPORT(Score, betaMargin3, betaMargins[3]);
	TUNE_EXPORT(Score, betaMargin4, betaMargins[4]);
	TUNE_EXPORT(Score, betaMargin5, betaMargins[5]);
	TUNE_EXPORT(Score, betaMargin6, betaMargins[6]);

	TUNE_EXPORT(Score, singularMargin, singularMargin);
	TUNE_EXPORT(Score, lateMoveFutility, lateMoveFutility);
}

// LazySMPThread

LazySMPThread::LazySMPThread() : searching(0), shouldQuit(0)
{
	memset( (void *)&commandData, 0, sizeof(commandData) );
}

void LazySMPThread::destroy()
{
	search.abort(1);
	shouldQuit = 1;
	commandEvent.signal();
	quitEvent.wait();
}

void LazySMPThread::work()
{
	SearchMode sm;
	sm.reset();
	search.mode = sm;

	for (;;)
	{
		commandEvent.wait();
		// search or quit!
		if ( shouldQuit )
			break;

		Depth depth;
		Score alpha, beta;
		const CommandData &c = commandData;
		depth = c.depth;
		alpha = c.alpha;
		beta = c.beta;
		search.mode.multiPV = c.multiPV;
		search.rootMoves = c.rootMoves;
		search.abortRequest = 0;
		search.aborting = 0;
		search.rootMoves.bestMove = mcNone;
		searching = 1;
		startedSearch.signal();

		assert( search.searchFlags & sfNoTimeout );
		search.root( depth, alpha, beta );

		searching = 0;
		doneSearch.signal();
	}
	quitEvent.signal();
}

void LazySMPThread::abort()
{
	search.abort(1);
	doneSearch.wait();
}

void LazySMPThread::start( Depth depth, Score alpha, Score beta, const Search &master )
{
	CommandData &cd = commandData;
	cd.depth = depth;
	cd.alpha = alpha;
	cd.beta = beta;
	cd.rootMoves = master.rootMoves;
	cd.multiPV = master.mode.multiPV;

	commandEvent.signal();
	startedSearch.wait();
}

}
