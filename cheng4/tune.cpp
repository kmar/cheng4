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
	51, 88, 63, 321, 309, 418, 1090,
	101, 337, 332, 540, 979, 9, 11, 2,
	9, 9, -9, 18, 12, 13, 15, 270,
	266, 356, 625, 895, 394, 2647, 69, 80,
	182, 125, 372, 432, 407, 530, 416, 361,
	253, -11, 415, 120, 2, 48, 50, -37,
	119, 194, 140, 166, 183, 277, 0, 0,
	-74, 55, 97, 15, 87, -118, 1014, 243,
	130, -18, -70, 33, 95, 263, 0, -15,
	22, 139, 312, 820, 0, -4, 17, -19,
	168, 369, 979, 158, 178, 440, 755, 1280,
	1851, -517, -319, -239, -194, -131, -95, -38,
	9, 79, -543, -498, -294, -157, -58, 41,
	54, 49, -7, -257, -145, -67, -8, 53,
	65, 82, 72, 37, 15, 6, 56, 326,
	184, -316, -256, -150, -50, 48, 136, 170,
	210, 228, 220, 216, 218, 151, 240, -8,
	68, 111, 97, 95, 115, 145, 181, 253,
	305, 359, 323, 295, 507, -74, -327, 34,
	112, 246, 310, 352, 402, 436, 494, 525,
	545, 577, 578, 528, 676, 115, 246, 336,
	349, 368, 375, 379, 384, 394, 363, 407,
	398, 400, 441, 442, 787, 992, 1596, 1739,
	1874, 1405, 1951, 924, 688, -927, 820, 666,
	858, -61, 191, 262, 463, 546, 629, 679,
	775, 846, 887, 947, 966, 999, 1000, 1001,
	914, 843, 797, 807, 750, 795, 815, 821,
	797, 855, 873, 923, 817, 1195, -15, 140,
	99, 86, 95, 53, 136, 36, 103, 81,
	139, 193, 201, 214, 442, 592, -2090, -353,
	-193, -28, -47, -22, -21, -56, 138, 65,
	98, 131, 175, 225, 254, 400, -246, 107,
	67, 30, 126, 98, -45, -73, -99, 17,
	11, 23, 36, 37, 63, 48, 13, 2,
	-6, 7, 23, 15, 23, -14, 0, -2,
	-9, 5, 16, 17, 19, -1, 0, -3,
	-11, -4, -2, 11, -5, 8, -3, -6,
	-11, -10, -11, 2, 16, 22, -12, 47,
	47, 51, 15, -3, 13, 44, 17, 40,
	34, 16, -11, -18, 5, 14, 11, 24,
	14, -1, -18, -12, -7, 2, 2, 12,
	5, -5, -11, -12, -2, -1, -9, 8,
	2, -7, -7, -1, 1, 1, -9, 10,
	2, -4, -4, 6, 9, 4, -12, -134,
	-113, -63, -42, 29, -179, 138, -46, -18,
	-27, -5, 11, -20, 76, -38, -35, -9,
	-3, 20, 23, 80, 86, 25, -18, 19,
	2, 15, 26, 3, 48, 13, 33, -5,
	-7, 10, 6, 19, 14, 19, 3, -15,
	1, -7, 3, 19, 7, 17, -2, -33,
	-9, -8, 6, 2, 10, -8, 0, -30,
	-11, -31, -2, 7, 2, -19, -74, -19,
	4, 7, 5, -3, 4, -37, -57, -2,
	7, 4, 13, -1, -3, -4, -4, 1,
	8, 18, 16, 17, 14, 2, 3, 5,
	6, 16, 17, 17, 17, 8, 4, 3,
	6, 18, 17, 18, 16, 11, 1, -21,
	-6, -3, 13, 15, 3, -1, -9, -35,
	-9, -12, -3, 0, -7, 12, 8, -65,
	-12, -15, -4, 8, 0, -9, -51, -55,
	-74, -69, -126, -90, -155, -2, -11, -41,
	-37, -40, -53, -28, -50, -78, -64, -41,
	-21, -16, 13, 35, 59, 13, -5, -19,
	-11, 0, 33, 1, -1, -4, -17, -9,
	-11, -9, 8, 11, -9, -12, 7, -6,
	8, 1, -12, -1, 5, 10, 16, 0,
	1, -6, -9, -2, 23, 19, 18, -7,
	-8, -14, -20, 3, -12, 11, -24, 11,
	16, 6, 10, 3, 2, -1, -8, 8,
	2, 9, 3, 8, 6, 3, -9, 2,
	15, 5, 10, 5, 26, 13, 9, -1,
	6, 10, 20, 21, 10, 5, -1, -21,
	-2, 13, 20, 7, 8, -3, -5, -9,
	0, 8, 2, 18, 4, 4, -11, -16,
	-6, -9, -1, -1, -3, 2, -38, -10,
	-5, 4, -5, -2, 3, -9, -5, -4,
	-21, -9, 5, 26, -5, 102, 152, -8,
	5, 12, 30, 16, 19, -18, 51, -22,
	5, 9, 22, 58, 65, 49, 15, -23,
	-18, 7, 20, 15, 37, 22, -11, -17,
	-37, -25, -13, -20, -30, -25, -20, -31,
	-20, -23, -16, -11, -31, -4, -41, -20,
	-33, -13, -13, -12, 3, -5, -73, -16,
	-11, -5, -1, 1, -1, -15, -10, 12,
	18, 20, 20, 16, 21, 22, 18, 25,
	27, 32, 31, 29, 19, 24, 24, 25,
	24, 26, 26, 19, 25, 22, 20, 24,
	29, 25, 25, 21, 23, 19, 16, 10,
	15, 18, 15, 10, 15, 13, 3, -1,
	1, 2, 0, -2, 1, 0, -10, -12,
	-8, -4, -7, -3, -11, -12, -21, -12,
	-9, -5, -5, -7, -9, -9, -25, -52,
	-42, -48, -73, -69, -45, 88, -19, -21,
	-40, -11, -19, -44, -46, -6, 42, -16,
	-19, -20, -5, -16, 49, 48, -6, 5,
	-16, -6, -9, -15, -14, -17, -15, -16,
	-10, -14, -14, -13, -21, -7, -6, -5,
	-3, -8, -13, -5, -1, 5, -3, -9,
	-10, -1, 4, 3, 13, 15, -21, -3,
	-9, -6, 1, 5, 16, -4, 32, 9,
	24, 25, 40, 42, 40, 26, 42, 7,
	30, 43, 51, 47, 36, 35, 40, 15,
	29, 42, 40, 47, 48, 19, 44, 6,
	26, 31, 40, 49, 33, 33, 31, 6,
	9, 21, 33, 29, 29, 25, 19, -8,
	-3, 13, 4, 10, 17, 16, 2, -18,
	-9, -12, -16, -5, -30, -48, -25, -22,
	-12, -21, -19, -24, -62, -57, -27, -499,
	209, -113, -341, 50, -80, 66, 70, 66,
	30, -142, -74, -93, -84, -81, 21, -89,
	-85, -109, 60, -79, -80, -28, -85, 24,
	34, -76, -99, -95, -87, -129, -201, -99,
	-104, -75, -217, -129, -154, -163, -107, -54,
	-23, -65, -117, -95, -73, -50, -41, -3,
	-74, -37, -108, -73, -68, -14, -5, -29,
	7, 1, -87, -2, -57, 17, 29, -126,
	-24, -13, -16, -2, 19, 19, -100, -17,
	21, 20, 25, 21, 33, 21, -24, 14,
	29, 30, 30, 36, 41, 37, 22, 3,
	21, 27, 30, 35, 41, 37, 7, -21,
	9, 19, 30, 30, 28, 18, -7, -22,
	-2, 9, 17, 22, 14, 3, -12, -23,
	-5, 2, -1, 2, 3, -6, -15, -32,
	-23, -14, -20, -38, -21, -21, -43
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
