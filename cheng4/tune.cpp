/*
You can use this program under the terms of either the following zlib-compatible license
or as public domain (where applicable)

  Copyright (C) 2012-2015 Martin Sedlak

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

#include "tune.h"
#include "eval.h"

#include <iostream>
#include <fstream>
#include <algorithm>

namespace cheng4
{

std::vector<Feature> features;
std::vector<i16> featureVector;

static const i16 optimizedFeatureVector[] = {
	111, 25, -10, 4, 154, 407, 740,
	31, -49, 45, 45, 156, 281, 404, 40,
	273, 344, -10, -48, 73, 79, 158, 55,
	315, 311, 404, 1042, 93, 338, 335, 541,
	1003, 12, 12, 5, 5, 7, -10, 18,
	11, 15, 15, 233, 186, 325, 569, 903,
	866, 2647, -39, 69, 43, 111, 405, 367,
	403, 527, 499, 358, 201, 4, 403, 119,
	2, 49, 115, 23, 120, 199, 207, 146,
	112, 85, 0, 0, -112, 55, 109, 35,
	92, 203, 1106, 235, 118, 28, 10, 75,
	235, 649, 0, -11, 47, 166, 353, 989,
	0, 69, -54, -116, -12, 369, 868, 174,
	176, 439, 756, 1269, 1852, -512, -329, -239,
	-180, -131, -95, -37, 13, 132, -462, -516,
	-293, -156, -58, 51, 53, 55, -31, -258,
	-159, -63, -7, 38, 65, 78, 54, 42,
	63, 88, 61, 319, 625, -315, -255, -147,
	-35, 74, 145, 194, 222, 234, 221, 182,
	214, 153, 244, 41, 68, 125, 82, 63,
	114, 145, 209, 252, 320, 388, 325, 372,
	507, -76, -327, -4, 113, 252, 309, 353,
	402, 438, 494, 525, 562, 577, 583, 529,
	638, 115, 214, 356, 382, 389, 375, 380,
	382, 363, 364, 388, 402, 406, 507, 587,
	785, 993, 1585, 1744, 1880, 1406, 1945, 926,
	688, -931, -3950, 2540, 1165, -61, 195, 266,
	463, 546, 629, 679, 774, 849, 918, 949,
	965, 993, 954, 952, 913, 877, 795, 806,
	801, 811, 822, 821, 888, 819, 875, 848,
	994, 950, -96, 141, -80, -60, 41, 53,
	81, 36, 102, 150, 138, 189, 149, 327,
	267, 695, -2087, -143, -103, -159, -58, -63,
	-21, -6, 171, 68, 106, 152, 174, 223,
	252, 454, -248, 116, 15, 38, 90, 92,
	-10, -145, -54, 6, -3, 31, 28, 41,
	61, 19, 11, 4, 2, 7, 18, 22,
	29, -6, 0, -4, 0, 6, 12, 20,
	19, 8, 5, -7, -5, -6, -5, 12,
	-6, 11, -1, -11, -8, -7, -8, -3,
	16, 24, -11, 49, 57, 50, 18, 6,
	21, 58, 37, 42, 35, 15, -10, -20,
	5, 17, 14, 25, 14, -3, -19, -14,
	-8, 2, 0, 14, 6, -5, -10, -10,
	-2, -1, -8, 9, 2, -5, -5, -1,
	-1, -1, -8, 12, 5, -2, 4, 4,
	8, 1, -11, -183, -135, -64, 7, 33,
	-127, 140, -174, -13, -25, -1, 27, -3,
	46, -15, -2, -27, -3, 12, 25, 72,
	73, 31, -30, -11, 6, 16, 25, 7,
	38, 10, 26, 0, 14, 9, 8, 13,
	12, 5, 8, -20, -7, -1, 9, 16,
	8, 13, -9, -1, -6, -10, 6, 5,
	10, -10, 2, -52, -18, -6, -3, -4,
	8, -6, -39, -7, -3, -1, 15, 4,
	-7, -6, -54, -1, 8, 4, 12, 9,
	0, -5, -10, -1, 4, 18, 21, 10,
	26, 5, 0, 10, 4, 17, 20, 16,
	21, 13, 9, 7, 0, 19, 15, 26,
	16, 11, 9, -21, -2, -2, 13, 15,
	3, 3, 0, -40, -9, -16, 1, 4,
	-2, 5, -1, -52, -12, -14, -4, -1,
	-7, 2, -19, -44, -47, -101, -40, -88,
	-123, -17, 35, -39, -17, -28, -52, -42,
	-88, -78, -64, -17, -4, -8, 5, 5,
	61, -4, 7, -23, -7, 2, 23, 4,
	-7, -3, -28, -25, 5, -11, 9, 4,
	-9, -5, -1, -8, -7, 2, -7, -3,
	-1, 5, 1, 1, -1, 1, -8, -1,
	9, 20, 15, 2, 24, -13, -8, 7,
	-13, 3, -4, 6, 12, 5, 16, 6,
	-5, 3, -6, 16, 5, 6, 0, 5,
	5, 10, -10, -3, 13, 8, 7, 9,
	25, 21, -2, 2, 9, 13, 21, 18,
	14, 7, 5, -9, -1, 10, 22, 16,
	7, 3, -8, -6, 5, 12, 4, 18,
	4, 1, -7, -6, -4, -5, 1, -2,
	4, 0, -30, -6, -12, 1, -4, -5,
	0, -10, -16, -4, 12, -9, 4, -5,
	0, 102, 118, 6, -18, 15, 27, 33,
	44, 50, 77, -12, 17, -4, 17, 61,
	55, 81, 40, -16, -15, 10, 24, 22,
	10, 23, 28, -34, -21, -9, -8, -12,
	-14, 7, -25, -31, -16, -15, -18, 1,
	-18, 5, -30, -39, -21, -12, -5, -13,
	-2, 4, -28, -12, -8, -6, 0, 1,
	-3, 1, -8, 13, 20, 19, 21, 22,
	22, 19, 18, 27, 28, 31, 32, 30,
	20, 23, 23, 26, 23, 27, 26, 23,
	23, 19, 22, 24, 29, 27, 26, 21,
	26, 22, 18, 12, 18, 18, 14, 12,
	15, 13, 5, -4, 1, -3, 0, -2,
	1, 2, -10, -17, -7, -2, -7, -6,
	-9, -8, -15, -13, -12, -5, -5, -10,
	-8, -9, -19, -27, -43, -40, -71, -68,
	-46, 137, 59, -20, -35, -6, -18, -24,
	-12, -5, 59, -20, -9, -19, -11, -8,
	56, 47, 18, -10, -8, -6, -16, -13,
	-2, 7, -3, -12, -10, -12, -11, -14,
	-13, 2, -20, -9, -3, -10, -12, -4,
	-6, 6, 5, -18, -6, 1, 6, 3,
	10, 33, -2, -10, -12, 4, 1, -10,
	7, 4, -7, 12, 23, 32, 37, 36,
	37, 23, 30, 13, 27, 40, 48, 46,
	29, 30, 41, 10, 29, 39, 46, 52,
	39, 28, 29, 8, 26, 37, 44, 46,
	34, 40, 33, 7, 13, 19, 33, 29,
	23, 25, 23, -8, -3, 10, 4, 15,
	15, 13, 1, -19, -10, -12, -13, -5,
	-29, -44, -15, -23, -21, -21, -18, -11,
	-52, -46, -24, -390, 40, 17, -332, -20,
	-104, 65, 80, 65, 32, -13, -74, -91,
	-1, 5, 26, 48, -85, -109, 60, -132,
	-78, -27, -1, 47, 34, -67, -185, -179,
	-129, -127, -202, -19, -35, -74, -217, -168,
	-153, -163, -132, -132, 14, -96, -118, -95,
	-104, -85, -55, -2, -105, -52, -110, -80,
	-68, -15, -9, -48, -9, 6, -79, -2,
	-59, 19, 24, -158, -28, -21, 1, -1,
	13, 19, -107, -45, 20, 23, 17, 37,
	45, 57, -35, 8, 29, 37, 40, 36,
	58, 51, 24, 0, 15, 34, 38, 39,
	41, 40, 12, -19, 11, 23, 32, 31,
	28, 18, -6, -29, -5, 10, 19, 21,
	14, 4, -11, -16, -8, 2, 1, 3,
	2, -5, -17, -42, -17, -12, -22, -35,
	-19, -23, -42
};

void addFeature(const char *name, i16 *ptr, int count)
{
	Feature feat;
	feat.name = name;
	feat.table = ptr;
	feat.size = count;
	feat.start = (int)featureVector.size();

	features.push_back(feat);

	for (int i=0; i<feat.size; i++)
		featureVector.push_back(ptr[i]);
}

bool saveFeatures(const char *filename)
{
	std::ofstream ofs( filename, std::ios::out );

	if (!ofs.is_open())
		return 0;

	ofs << "\t";

	for (size_t i=0; i<featureVector.size(); i++)
	{
		if ((i & 7) == 7)
			ofs << std::endl << "\t";

		ofs << featureVector[i];

		if (i+1 < featureVector.size())
			ofs << ", ";
	}

	ofs << std::endl;
	return 1;
}

#define ADD_SINGLE_FEATURE(x) addFeature(#x, &x)

void extractFeatures()
{
	ADD_SINGLE_FEATURE(disconnectedPawn);
	ADD_SINGLE_FEATURE(disconnectedPawnEg);

	addFeature("connectedPasserOpening", connectedPasserOpening+1, 6);
	addFeature("connectedPasserEndgame", connectedPasserEndgame+1, 6);

	addFeature("kingOpenFile", kingOpenFile+1, 3);
	addFeature("kingOpenFileEndgame", kingOpenFileEg+1, 3);

	ADD_SINGLE_FEATURE(rookBehindPasserOpening);
	ADD_SINGLE_FEATURE(rookBehindPasserEndgame);

	addFeature("materialOpening", PSq::materialTables[phOpening]+1, 5);
	addFeature("materialEndgame", PSq::materialTables[phEndgame]+1, 5);

	addFeature("safetyScale", safetyScale+1, 5);
	addFeature("safetyScaleEg", safetyScaleEg+1, 5);

	ADD_SINGLE_FEATURE(shelterFront1);
	ADD_SINGLE_FEATURE(shelterFront2);

	ADD_SINGLE_FEATURE(bishopPairOpening);
	ADD_SINGLE_FEATURE(bishopPairEndgame);

	ADD_SINGLE_FEATURE(trappedBishopOpening);
	ADD_SINGLE_FEATURE(trappedBishopEndgame);

	ADD_SINGLE_FEATURE(unstoppablePasser);

	ADD_SINGLE_FEATURE(doubledPawnOpening);
	ADD_SINGLE_FEATURE(doubledPawnEndgame);

	ADD_SINGLE_FEATURE(isolatedPawnOpening);
	ADD_SINGLE_FEATURE(isolatedPawnEndgame);

	ADD_SINGLE_FEATURE(knightHangingOpening);
	ADD_SINGLE_FEATURE(knightHangingEndgame);

	ADD_SINGLE_FEATURE(bishopHangingOpening);
	ADD_SINGLE_FEATURE(bishopHangingEndgame);

	ADD_SINGLE_FEATURE(rookHangingOpening);
	ADD_SINGLE_FEATURE(rookHangingEndgame);

	ADD_SINGLE_FEATURE(rookOnOpenOpening);
	ADD_SINGLE_FEATURE(rookOnOpenEndgame);

	ADD_SINGLE_FEATURE(queenHangingOpening);
	ADD_SINGLE_FEATURE(queenHangingEndgame);

	ADD_SINGLE_FEATURE(kingPasserSupportBase);
	ADD_SINGLE_FEATURE(kingPasserSupportScale);

	addFeature("outpostBonusFile", outpostBonusFile, 8);
	addFeature("outpostBonusRank", outpostBonusRank, 8);

	ADD_SINGLE_FEATURE(pawnRaceAdvantageEndgame);

	addFeature("passerScaleImbalance", passerScaleImbalance, 1);
	addFeature("passerScaleBlocked", passerScaleBlocked, 1);

	addFeature("candPasserOpening", candPasserOpening+1, 6);
	addFeature("candPasserEndgame", candPasserEndgame+1, 6);

	addFeature("passerOpening", passerOpening+1, 6);
	addFeature("passerEndgame", passerEndgame+1, 6);

	addFeature("knightMobilityOpening", knightMobility[phOpening], 9);
	addFeature("knightMobilityEndgame", knightMobility[phEndgame], 9);

	addFeature("bishopMobilityOpening", bishopMobility[phOpening], 14);
	addFeature("bishopMobilityEndgame", bishopMobility[phEndgame], 14);

	addFeature("rookMobilityOpening", rookMobility[phOpening], 15);
	addFeature("rookMobilityEndgame", rookMobility[phEndgame], 15);

	addFeature("queenMobilityOpening", queenMobility[phOpening], 28);
	addFeature("queenMobilityEndgame", queenMobility[phEndgame], 28);

	addFeature("goodBishopOpening", goodBishopOpening, 17);
	addFeature("goodBishopEndgame", goodBishopEndgame, 17);

	// and finally piece-square tables

	addFeature("pawnPsqOpening", PSq::pawnTables[phOpening] + 8, 64-2*8);
	addFeature("pawnPsqEndgame", PSq::pawnTables[phEndgame] + 8, 64-2*8);

	addFeature("knightPsqOpening", PSq::knightTables[phOpening], 64);
	addFeature("knightPsqEndgame", PSq::knightTables[phEndgame], 64);

	addFeature("bishopPsqOpening", PSq::bishopTables[phOpening], 64);
	addFeature("bishopPsqEndgame", PSq::bishopTables[phEndgame], 64);

	addFeature("rookPsqOpening", PSq::rookTables[phOpening], 64);
	addFeature("rookPsqEndgame", PSq::rookTables[phEndgame], 64);

	addFeature("queenPsqOpening", PSq::queenTables[phOpening], 64);
	addFeature("queenPsqEndgame", PSq::queenTables[phEndgame], 64);

	addFeature("kingPsqOpening", PSq::kingTables[phOpening], 64);
	addFeature("kingPsqEndgame", PSq::kingTables[phEndgame], 64);

	// load actual optimized values

	size_t optsz = sizeof(optimizedFeatureVector) / sizeof(i16);

	optsz = std::min(optsz, featureVector.size());

	for (size_t i=0; i<optsz; i++)
		featureVector[i] = optimizedFeatureVector[i];

	for (size_t i=0; i<features.size(); i++)
	{
		const Feature &ft = features[i];

		for (int j=0; j<ft.size; j++)
			ft.table[j] = featureVector[ft.start + j];
	}

	// re-init psq to bake material into them
	PSq::init();
}

#undef ADD_SINGLE_FEATURE

}

#ifdef USE_TUNING

#include <iostream>

namespace cheng4
{

// TunableParams

TunableParams *TunableParams::inst = 0;

TunableParams *TunableParams::get()
{
	// note: not thread-safe
	if ( !inst )
		inst = new TunableParams;
	return inst;
}

std::vector< TunableBase * > TunableParams::params;

void TunableParams::addParam( TunableBase *param )
{
	params.push_back( param );
}

bool TunableParams::setParam( const char *name, const char *value )
{
	for ( size_t i=0; i<params.size(); i++ )
	{
		if ( params[i]->name() == name )
		{
			params[i]->set( value );
			return 1;
		}
	}
	return 0;
}

// dump params
void TunableParams::dump()
{
	for ( size_t i=0; i<params.size(); i++ )
	{
		std::cout << params[i]->name() << " = " << params[i]->get() << std::endl;
	}
}

size_t TunableParams::paramCount()
{
	return params.size();
}

const TunableBase *TunableParams::getParam( size_t index )
{
	return params[index];
}

TunableBase *TunableParams::findParam( const char *name )
{
	for (size_t i=0; i<params.size(); i++)
	{
		if (params[i]->name() == name) {
			return params[i];
		}
	}
	return 0;
}

}

#endif
