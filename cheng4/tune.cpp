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
	63, 20, 65, 113, 48, 320, 740,
	2221, -72, 0, 21, 131, 281, 260, 70,
	273, 342, -14, -40, 112, 36, 87, 60,
	318, 309, 416, 1089, 98, 337, 332, 540,
	979, 12, 11, 3, 8, 7, -9, 18,
	12, 13, 16, 253, 234, 356, 624, 901,
	394, 2649, -40, 45, 127, 125, 406, 402,
	409, 527, 415, 360, 254, -12, 401, 121,
	2, 44, 34, -38, 118, 198, 142, 143,
	153, 248, 0, 0, -77, 55, 110, 15,
	86, -118, 1014, 235, 121, -14, -29, 36,
	147, 264, 0, 4, 38, 140, 336, 820,
	0, -3, 18, -70, -12, 369, 979, 174,
	176, 439, 756, 1253, 1851, -513, -319, -239,
	-194, -131, -95, -37, 10, 132, -542, -497,
	-295, -157, -58, 41, 52, 49, -32, -257,
	-145, -83, -20, 53, 65, 82, 69, 38,
	15, 9, 61, 327, 176, -316, -255, -147,
	-50, 49, 136, 169, 210, 228, 220, 213,
	215, 152, 241, -8, 68, 111, 96, 95,
	115, 145, 183, 253, 305, 387, 323, 377,
	507, -74, -324, 32, 112, 252, 309, 352,
	402, 436, 494, 525, 545, 577, 579, 528,
	645, 115, 253, 338, 349, 368, 375, 380,
	383, 363, 364, 407, 398, 406, 439, 586,
	784, 994, 1596, 1743, 1875, 1405, 1952, 925,
	688, -927, 389, 666, 859, -63, 194, 263,
	463, 546, 628, 679, 775, 846, 919, 947,
	965, 995, 1001, 953, 913, 841, 795, 806,
	750, 726, 816, 821, 774, 819, 875, 848,
	818, 1198, -14, 141, 99, 82, 96, 53,
	137, 36, 105, 81, 139, 193, 204, 214,
	442, 591, -2091, -191, -191, -28, -47, -23,
	-21, -56, 138, 65, 98, 136, 175, 223,
	253, 401, -250, 115, 62, 22, 127, 99,
	-62, -82, -118, 14, 4, 14, 29, 30,
	64, 31, 13, 4, -3, 8, 20, 14,
	20, -11, 1, -4, -8, 6, 15, 19,
	17, 4, 1, -7, -11, -5, -4, 11,
	-6, 11, -3, -8, -9, -9, -12, 4,
	16, 25, -13, 52, 49, 54, 18, -2,
	13, 42, 21, 43, 34, 15, -12, -20,
	4, 9, 12, 25, 14, -3, -18, -13,
	-10, 0, 2, 15, 6, -4, -10, -11,
	-2, -1, -8, 10, 2, -7, -6, -1,
	0, 0, -9, 12, 2, -3, -2, 6,
	9, 3, -11, -132, -112, -61, -41, 32,
	-180, 139, -71, -14, -25, -4, 13, -15,
	77, -48, -34, -11, -2, 21, 25, 79,
	88, 25, -18, 20, 3, 16, 28, 6,
	49, 13, 33, -5, -4, 11, 6, 20,
	15, 20, 3, -16, 0, -5, 4, 20,
	8, 18, -1, -33, -9, -9, 6, 1,
	11, -7, 0, -31, -11, -30, -1, 5,
	2, -16, -80, -19, 4, 6, 5, -4,
	4, -38, -54, -1, 8, 4, 13, -1,
	-4, -4, -3, 0, 7, 18, 15, 17,
	14, 4, 3, 5, 4, 16, 16, 16,
	18, 9, 5, 3, 6, 18, 18, 18,
	16, 11, 1, -21, -6, -3, 13, 15,
	4, -1, -8, -35, -9, -13, -3, 0,
	-7, 12, 8, -65, -12, -15, -4, 8,
	0, -10, -51, -64, -75, -68, -128, -89,
	-155, -1, -10, -40, -37, -40, -52, -27,
	-41, -77, -64, -41, -20, -16, 13, 36,
	61, 11, -5, -18, -11, -2, 34, 3,
	-2, -4, -17, -10, -11, -8, 8, 11,
	-9, -13, 6, -7, 8, 1, -13, -1,
	4, 10, 16, -1, 1, -7, -9, -3,
	24, 19, 18, -7, -8, -14, -20, 3,
	-9, 10, -20, 9, 15, 6, 10, 3,
	2, -1, -9, 8, 2, 8, 3, 7,
	5, 2, -9, 2, 15, 5, 10, 5,
	26, 13, 9, -1, 6, 10, 20, 21,
	12, 5, -1, -21, -3, 13, 21, 7,
	8, -3, -5, -10, -1, 8, 2, 18,
	4, 4, -11, -16, -6, -9, -1, -1,
	-2, 2, -38, -10, -5, 4, -5, -2,
	3, -10, -5, -7, -24, -8, 3, 26,
	-6, 102, 118, -11, 5, 13, 30, 17,
	20, -18, 51, -22, 5, 7, 22, 60,
	70, 49, 8, -26, -18, 7, 20, 12,
	41, 22, -12, -19, -37, -25, -13, -20,
	-30, -25, -20, -31, -20, -23, -15, -11,
	-30, -1, -38, -20, -33, -13, -13, -12,
	3, -4, -66, -16, -11, -5, -1, 1,
	-1, -15, -8, 12, 18, 19, 19, 16,
	21, 22, 19, 25, 27, 31, 31, 29,
	19, 23, 24, 25, 24, 26, 25, 19,
	24, 22, 21, 24, 29, 25, 25, 21,
	23, 20, 17, 10, 15, 18, 14, 10,
	15, 14, 3, -2, 1, 2, 0, -2,
	0, 0, -10, -12, -8, -4, -7, -4,
	-11, -12, -21, -12, -9, -5, -5, -7,
	-9, -9, -25, -51, -44, -48, -71, -69,
	-47, 88, -15, -20, -39, -14, -19, -38,
	-45, -6, 35, -16, -19, -19, -2, -16,
	49, 39, -6, 5, -16, -6, -9, -17,
	-14, -17, -15, -16, -10, -14, -14, -13,
	-21, -8, -5, -5, -3, -8, -13, -6,
	-1, 5, -3, -11, -10, -1, 3, 3,
	13, 15, -26, -1, -9, -6, 1, 5,
	16, -4, 32, 9, 24, 24, 37, 39,
	39, 23, 39, 8, 30, 43, 51, 43,
	36, 33, 40, 15, 29, 42, 40, 46,
	46, 19, 37, 6, 27, 31, 40, 49,
	31, 33, 32, 6, 10, 21, 33, 29,
	28, 25, 19, -8, -3, 13, 4, 10,
	17, 16, 2, -18, -9, -12, -17, -5,
	-30, -44, -24, -22, -12, -21, -19, -24,
	-62, -58, -27, -498, 209, -113, -341, -21,
	-103, 65, 71, 65, 30, -141, -74, -91,
	2, -81, 25, -88, -85, -109, 60, -77,
	-79, -27, -85, 49, 35, -68, -97, -98,
	-90, -127, -201, -100, -103, -75, -217, -129,
	-153, -163, -108, -53, -22, -65, -118, -95,
	-73, -50, -48, -2, -74, -36, -98, -73,
	-68, -14, -5, -28, 7, 1, -86, -4,
	-57, 17, 29, -127, -27, -13, -15, -1,
	20, 15, -105, -22, 21, 22, 28, 21,
	33, 21, -24, 13, 29, 32, 32, 36,
	42, 35, 20, 0, 22, 29, 32, 36,
	41, 36, 6, -22, 9, 21, 32, 30,
	27, 17, -8, -22, -2, 10, 19, 22,
	13, 3, -12, -24, -6, 2, 0, 2,
	3, -6, -16, -33, -23, -13, -18, -38,
	-21, -21, -44
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
