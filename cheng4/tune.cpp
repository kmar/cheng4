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
	61, 322, 310, 414, 1100, 103, 338,
	333, 543, 978, -1, 5, 3, 5, 7,
	-2, 2, 3, 3, 4, 272, 249, 352,
	609, 798, 545, 2641, 84, 78, 184, 125,
	322, 406, 373, 500, 507, 410, 253, -11,
	396, 120, 2, 50, 50, -31, 124, 203,
	163, 163, 183, 349, 0, 0, -74, 55,
	98, 39, 86, 10, 1011, 243, 133, -17,
	-70, 34, 25, 253, 0, -16, 20, 123,
	309, 816, 0, -5, 9, -13, 247, 470,
	1092, 158, 178, 434, 755, 1280, 1851, -445,
	-322, -249, -195, -131, -95, -38, 40, 77,
	-552, -497, -271, -158, -59, 32, 54, 50,
	-6, -216, -149, -58, -6, 57, 66, 81,
	76, 66, 17, 3, -29, 315, 365, -324,
	-256, -133, -66, 48, 134, 169, 212, 225,
	236, 221, 240, 156, 238, -8, 68, 112,
	85, 96, 115, 146, 183, 220, 304, 349,
	271, 289, 378, -167, -243, 33, 110, 245,
	292, 331, 382, 436, 494, 522, 572, 583,
	609, 535, 711, 179, 273, 332, 313, 366,
	364, 361, 382, 395, 395, 410, 398, 502,
	446, 441, 578, 647, 627, 649, 672, 694,
	722, 746, 707, 792, 817, 834, 842, -64,
	190, 262, 527, 548, 629, 678, 722, 782,
	821, 906, 965, 999, 1002, 1081, 1016, 1036,
	943, 917, 917, 893, 928, 933, 945, 1166,
	1010, 1130, 1134, 1082, -17, 87, 201, 91,
	93, 110, 136, 33, 118, 81, 139, 193,
	198, 213, 445, 590, -1838, -357, -262, -79,
	-47, -22, -21, -33, 138, 65, 129, 131,
	175, 224, 253, 398, 121, 111, 66, 34,
	149, 84, -45, -34, -100, 23, 17, 22,
	40, 34, 85, 68, 31, 3, 4, 8,
	20, 21, 25, -9, 8, 0, -6, 5,
	15, 14, 21, -7, -1, 2, -7, -3,
	4, 11, -9, 5, -1, -3, -11, -13,
	-11, 0, 14, 17, -12, 47, 46, 54,
	13, -3, 12, 46, 12, 39, 34, 16,
	-12, -17, 5, 12, 8, 23, 12, -1,
	-17, -12, -6, 3, 3, 11, 5, -5,
	-11, -11, -2, -1, -8, 5, 1, -7,
	-8, -1, 1, 1, -8, 8, 1, -3,
	-3, 7, 9, 4, -11, -143, -111, -64,
	-16, 28, -162, -8, -47, -25, -30, -10,
	23, -34, 107, -39, -20, -10, 0, 20,
	23, 94, 94, 37, -17, 23, 2, 15,
	31, 3, 51, 0, 40, -2, -9, 9,
	3, 21, 19, 35, 3, -14, 2, -7,
	3, 22, 4, 17, -1, -41, -8, -5,
	7, 5, 10, 1, 1, -37, -13, -32,
	2, 6, -1, -11, -58, -22, 5, 6,
	8, 0, 8, -33, -54, 0, 10, 5,
	17, 4, -1, 1, -2, 1, 7, 18,
	17, 20, 17, 8, 5, 4, 7, 18,
	21, 18, 20, 15, 9, -1, 7, 18,
	20, 20, 19, 16, 1, -22, -7, -5,
	13, 13, 2, -4, -12, -31, -10, -14,
	-5, -2, -8, 8, 5, -64, -12, -16,
	-5, 7, -1, -12, -51, -57, -74, -67,
	-128, -88, -158, -25, -11, -36, -48, -41,
	-66, -21, -35, -58, -64, -42, -22, -19,
	15, 37, 74, 27, -1, -15, -16, 0,
	34, 8, -1, -1, -10, -15, -18, -10,
	12, 12, -8, -17, 13, -9, 8, 1,
	-12, 2, 6, 13, 14, -1, 2, -5,
	-8, -1, 25, 17, 19, -6, -7, -14,
	-20, 2, -10, 9, -22, 10, 16, 6,
	10, 4, 2, 0, -6, 8, 4, 7,
	3, 7, 7, 3, -9, 1, 15, 5,
	10, 5, 27, 14, 9, -2, 7, 9,
	20, 22, 13, 6, 3, -19, -1, 12,
	20, 8, 8, -2, -5, -9, 1, 9,
	4, 18, 4, 0, -11, -12, -7, -9,
	-1, -1, -4, 3, -44, -7, -4, 7,
	-5, -2, 1, -9, 2, 16, -12, 0,
	29, 22, 35, 97, 101, 0, 17, 30,
	45, 37, 71, -21, 66, -11, 20, 24,
	40, 81, 116, 109, 34, -19, -11, 6,
	27, 15, 63, 23, -4, -15, -33, -20,
	-4, -22, -25, -16, -20, -25, -18, -20,
	-11, -9, -29, -2, -45, -17, -33, -12,
	-10, -7, 4, -11, -83, -16, -9, -5,
	0, 2, -6, -23, -11, 10, 16, 18,
	17, 15, 20, 21, 17, 23, 26, 30,
	31, 28, 22, 25, 23, 25, 24, 26,
	26, 21, 27, 25, 22, 25, 30, 26,
	25, 22, 25, 22, 18, 10, 15, 18,
	14, 12, 15, 15, 4, -3, 1, 1,
	0, -2, 2, 0, -10, -14, -9, -4,
	-7, -5, -10, -13, -21, -12, -10, -6,
	-5, -8, -4, -7, -27, -46, -29, -47,
	-43, -47, -7, 87, -28, -22, -37, -4,
	1, -37, 22, -9, 41, -10, 1, -5,
	7, 7, 62, 49, 1, 12, -15, -8,
	-1, -6, -8, -16, -5, -16, -10, -9,
	-7, -5, -11, -7, 3, -5, -4, -1,
	-10, -2, 2, 15, 7, -14, -10, -2,
	5, 3, 14, 3, -30, -4, -6, -8,
	1, 8, 4, -22, 43, 4, 18, 19,
	30, 37, 41, 23, 37, 6, 26, 37,
	45, 50, 47, 31, 40, 15, 23, 37,
	39, 50, 59, 38, 49, 2, 24, 27,
	36, 48, 40, 41, 41, 6, 8, 18,
	31, 26, 32, 29, 24, -8, -2, 8,
	5, 10, 18, 12, -4, -17, -9, -9,
	-16, -5, -30, -46, -27, -21, -13, -21,
	-19, -32, -62, -58, -37, -498, 204, -114,
	-86, 47, -93, 64, 21, 62, 30, -143,
	-73, -94, -85, -86, 21, -87, -84, -86,
	60, -82, -84, -50, -87, -90, 101, -17,
	-89, -95, -87, -37, -206, -100, -104, -29,
	-106, -112, -87, -114, -147, -54, 25, -27,
	-80, -62, -22, -18, -42, 5, -46, -18,
	-77, -53, -45, -3, 0, -61, 7, 6,
	-71, 1, -63, 18, 21, -128, -21, -10,
	-16, -5, 24, 22, -111, -13, 25, 26,
	27, 25, 41, 20, -44, 17, 34, 35,
	34, 39, 41, 37, 23, 4, 21, 28,
	29, 35, 43, 38, 7, -22, 10, 19,
	29, 29, 27, 18, -9, -24, -2, 9,
	17, 21, 13, 3, -14, -25, -3, 3,
	-1, 4, 4, -4, -16, -32, -23, -14,
	-22, -41, -22, -21, -47
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
