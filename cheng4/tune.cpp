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
	33, 84, 63, 321, 309, 415, 1106,
	103, 337, 332, 542, 973, 8, 11, 2,
	11, 9, -7, 19, 13, 12, 14, 253,
	249, 353, 626, 894, 394, 2648, 85, 80,
	184, 125, 322, 432, 374, 531, 427, 358,
	253, -11, 365, 119, 2, 44, 51, -36,
	119, 176, 139, 163, 183, 277, 0, 0,
	-74, 55, 98, 16, 85, -112, 1011, 243,
	130, -18, -70, 34, 57, 262, 0, -15,
	20, 123, 311, 818, 0, -4, 9, -18,
	168, 369, 980, 158, 178, 434, 755, 1280,
	1851, -445, -286, -239, -194, -131, -95, -38,
	9, 76, -551, -498, -271, -158, -58, 35,
	54, 49, -7, -207, -130, -67, -6, 57,
	66, 82, 73, 28, 17, 4, -28, 325,
	187, -323, -256, -150, -66, 48, 134, 169,
	210, 226, 220, 215, 239, 153, 240, -9,
	68, 111, 85, 95, 115, 146, 181, 253,
	304, 350, 271, 291, 506, -157, -242, 33,
	111, 245, 292, 352, 382, 436, 494, 523,
	549, 577, 578, 529, 676, 185, 309, 359,
	349, 367, 364, 362, 384, 394, 362, 407,
	398, 399, 441, 441, 579, 672, 1124, 1739,
	1873, 705, 1949, 747, 709, -928, 818, 832,
	858, -61, 190, 263, 527, 546, 629, 678,
	776, 782, 887, 947, 966, 999, 1000, 1081,
	957, 926, 860, 806, 830, 798, 817, 822,
	898, 952, 919, 923, 927, 1194, -15, 89,
	103, 86, 93, 110, 136, 35, 118, 81,
	139, 193, 203, 214, 444, 591, -2090, -356,
	-262, -28, -47, -22, -21, -56, 138, 65,
	129, 131, 175, 225, 254, 400, -246, 108,
	66, 30, 126, 98, -46, -65, -99, 20,
	13, 23, 35, 37, 70, 53, 16, 5,
	-3, 7, 26, 16, 22, -12, 0, 0,
	-9, 6, 16, 16, 18, -5, -2, -1,
	-10, -4, 1, 11, -6, 7, -3, -4,
	-11, -12, -10, 1, 15, 20, -12, 47,
	48, 54, 17, 0, 14, 47, 16, 40,
	35, 17, -10, -18, 5, 12, 9, 23,
	13, 0, -18, -12, -7, 2, 2, 11,
	5, -5, -11, -12, -2, -1, -9, 7,
	1, -8, -8, -1, 0, 1, -8, 9,
	2, -3, -3, 6, 9, 4, -12, -136,
	-114, -63, -15, 29, -161, 138, -46, -13,
	-28, -2, 10, -31, 76, -38, -20, -10,
	-1, 20, 28, 78, 94, 37, -13, 22,
	2, 14, 24, 3, 49, 8, 33, -3,
	-8, 9, 5, 19, 14, 20, 1, -14,
	1, -8, 3, 20, 6, 18, -2, -35,
	-11, -7, 7, 2, 10, -8, -1, -36,
	-11, -33, -3, 3, 0, -14, -90, -21,
	5, 7, 4, -3, 4, -34, -54, -1,
	8, 4, 14, 2, -2, -2, -4, 1,
	8, 19, 16, 17, 14, 5, 1, 6,
	6, 17, 18, 17, 18, 8, 6, 3,
	7, 18, 20, 18, 16, 12, 1, -22,
	-7, -3, 13, 15, 3, -1, -10, -34,
	-9, -13, -4, 0, -8, 12, 6, -62,
	-8, -15, -5, 8, -1, -10, -51, -55,
	-74, -70, -128, -87, -159, -2, -11, -41,
	-38, -40, -50, -29, -50, -73, -55, -38,
	-22, -15, 14, 38, 59, 16, -2, -18,
	-15, 0, 31, 1, -1, -4, -17, -9,
	-14, -10, 8, 12, -8, -13, 10, -6,
	8, -1, -12, -1, 5, 11, 16, -1,
	1, -6, -8, -2, 26, 19, 21, -7,
	-7, -14, -19, 2, -13, 10, -33, 11,
	17, 7, 11, 4, 3, 2, -5, 8,
	2, 8, 3, 8, 6, 4, -9, 2,
	15, 5, 10, 5, 27, 14, 8, -1,
	8, 10, 20, 21, 10, 6, 2, -20,
	-2, 13, 20, 7, 7, -3, -6, -9,
	0, 8, 2, 18, 4, 4, -11, -13,
	-6, -9, -1, -1, -3, 3, -40, -10,
	-6, 4, -5, -2, 2, -9, 2, 5,
	-10, 12, 20, 26, 34, 133, 151, -5,
	5, 19, 30, 19, 19, -36, 66, -22,
	5, 9, 21, 56, 63, 46, 3, -22,
	-18, 7, 20, 14, 32, 22, -11, -14,
	-37, -20, -13, -21, -30, -25, -20, -29,
	-20, -23, -16, -12, -34, -4, -41, -20,
	-33, -13, -13, -12, 1, -5, -75, -16,
	-10, -5, -1, 1, -5, -19, -9, 11,
	17, 19, 19, 17, 21, 21, 17, 24,
	27, 31, 31, 29, 20, 24, 23, 25,
	24, 26, 26, 20, 26, 22, 20, 25,
	30, 26, 25, 21, 25, 22, 17, 10,
	15, 18, 15, 11, 15, 15, 3, -2,
	1, 2, -1, -2, 1, 1, -11, -14,
	-9, -4, -7, -5, -13, -13, -21, -12,
	-9, -5, -5, -7, -6, -8, -27, -45,
	-29, -31, -42, -46, -6, 135, 4, -22,
	-37, -11, -7, -37, -45, 7, 41, -18,
	-14, -20, 2, -16, 43, 48, -1, 5,
	-16, -6, -6, -14, -15, -17, -14, -16,
	-9, -11, -14, -12, -20, -7, -6, -5,
	-3, -4, -12, -5, -1, 8, 0, -10,
	-10, -1, 4, 3, 14, 19, -30, -3,
	-9, -7, 1, 8, 16, -16, 43, 7,
	22, 23, 36, 41, 40, 23, 42, 7,
	28, 43, 50, 50, 40, 35, 40, 15,
	29, 43, 40, 50, 51, 23, 44, 6,
	28, 27, 40, 48, 33, 34, 38, 6,
	9, 20, 35, 30, 29, 25, 24, -8,
	-3, 10, 4, 10, 17, 16, 1, -18,
	-9, -12, -16, -5, -30, -48, -27, -20,
	-12, -21, -19, -28, -62, -58, -37, -498,
	209, -113, -85, 48, -92, 65, 47, 63,
	30, -143, -72, -94, -85, -85, 20, -90,
	-84, -109, 60, -80, -83, -29, -85, 18,
	101, -60, -95, -94, -88, -129, -205, -98,
	-104, -74, -106, -121, -86, -163, -107, -55,
	-24, -66, -116, -94, -73, -50, -41, 5,
	-77, -37, -108, -73, -68, -14, -5, -29,
	7, 1, -72, 1, -57, 17, 29, -127,
	-24, -13, -16, -1, 22, 19, -111, -18,
	22, 25, 25, 25, 34, 17, -40, 17,
	32, 32, 32, 36, 40, 35, 21, 3,
	22, 28, 30, 35, 41, 37, 8, -22,
	10, 19, 29, 29, 27, 18, -7, -23,
	-2, 9, 17, 22, 14, 3, -12, -26,
	-5, 2, -1, 3, 4, -5, -15, -31,
	-23, -14, -21, -42, -21, -21, -45
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
