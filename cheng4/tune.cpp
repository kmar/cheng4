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
	-15, -24, -41, -45, -19, -24, 3,
	35, 49, 94, 174, 257, 428, 544, 644,
	1011, 1153, 1503, 1832, 1841, 2194, 2985, 2258,
	2802, 2610, 2247, 1, 261, 0, -1, 95,
	25, 70, 27, 118, 283, 935, 452, -59,
	26, 43, 154, 259, 515, 64, 211, 305,
	17, 6, 86, 83, 151, 57, 308, 302,
	378, 858, 89, 339, 340, 550, 1034, 10,
	11, 6, 5, 8, -9, 19, 12, 15,
	16, 145, 111, 326, 568, 1235, 864, 2234,
	14, 64, 20, 104, 372, 366, 362, 531,
	545, 331, 259, 28, 378, 128, 2, 49,
	-102, -6, 130, 185, 232, 200, 183, 146,
	0, 0, -89, 22, 133, 3, -5, 36,
	897, 247, 122, -5, -34, 53, 173, 346,
	0, 11, 41, 188, 337, 940, 0, -88,
	-82, -124, -9, 298, 948, 181, 158, 427,
	684, 1270, 1962, -576, -376, -274, -198, -128,
	-101, -61, -8, 79, -464, -513, -264, -151,
	-57, 50, 68, 56, -30, -262, -172, -82,
	-40, 14, 46, 64, 70, 85, 68, 79,
	123, 289, 531, -372, -293, -161, -35, 77,
	148, 194, 208, 232, 227, 226, 215, 218,
	195, -75, 12, 60, 50, 65, 111, 146,
	210, 283, 333, 384, 503, 540, 588, 139,
	-328, 5, 119, 253, 306, 351, 402, 440,
	494, 545, 562, 576, 585, 529, 597, 116,
	289, 323, 330, 360, 368, 380, 354, 363,
	366, 426, 400, 475, 528, 592, 790, 1036,
	1539, 1726, 1875, 1446, 1907, 932, 693, -930,
	-3961, 3914, 1171, -482, 200, 269, 468, 540,
	631, 680, 775, 850, 917, 915, 959, 939,
	938, 951, 915, 873, 860, 850, 885, 924,
	875, 967, 925, 896, 1093, 1034, 1043, 350,
	17, -130, 19, 22, 43, 16, 48, -28,
	81, 86, 138, 173, 188, 279, 177, 550,
	-2078, -144, -67, -120, -77, -62, -20, 7,
	126, 70, 105, 156, 171, 237, 253, 460,
	-131, 89, 51, 71, 99, 70, -7, -144,
	-56, 15, 11, 33, 37, 58, 96, 38,
	17, 2, 11, 10, 20, 30, 26, 10,
	5, -5, 2, 9, 16, 20, 19, 14,
	1, -9, -8, -6, -8, 9, -1, 14,
	2, -7, -9, -10, -15, -1, 11, 27,
	-6, 52, 55, 52, 18, 9, 15, 39,
	37, 37, 33, 16, -9, -18, 10, 19,
	17, 30, 17, 3, -15, -9, -4, 8,
	4, 15, 8, -6, -12, -11, -3, -2,
	-7, 10, 4, -4, -8, -3, -1, -5,
	-7, 12, 4, -2, -7, 8, 7, 0,
	-11, -97, -31, -97, -19, 30, -163, 194,
	-119, -27, -31, 25, 13, 22, 29, 1,
	-17, -33, 10, 16, 25, 65, 74, 26,
	-1, 7, 5, 10, 27, 6, 43, 10,
	25, -2, -1, 10, 7, 11, 19, 24,
	-1, -20, -2, -6, 14, 18, 6, 13,
	-4, -37, -21, -10, 2, 5, 7, -1,
	1, -18, -13, -19, -13, 9, -5, -10,
	11, -14, 0, 6, 5, 11, -1, -3,
	-37, 5, 11, 5, 11, 8, -2, 2,
	4, 3, 9, 26, 21, 17, 23, 5,
	9, 12, 7, 22, 25, 18, 22, 19,
	14, 3, 9, 21, 22, 26, 19, 12,
	10, -17, -6, 1, 18, 18, 4, 0,
	1, -30, -9, -9, 2, 3, 2, 0,
	-2, -58, -4, -21, 1, -4, -3, -5,
	-26, -22, -48, -93, -76, -90, -118, -18,
	-82, -31, -12, -22, -46, -17, -41, -34,
	-54, -28, -11, 1, -2, 10, 69, 10,
	-1, -31, -8, 1, 25, 10, 6, -9,
	-24, -17, -13, -2, 17, -2, -9, -3,
	-2, -14, 7, -6, -9, -3, 8, 6,
	18, 5, 0, -5, -12, -3, 8, 20,
	10, -1, -3, -13, -18, -2, -15, -1,
	-3, 2, 13, 9, 12, 11, -3, 5,
	-5, 9, 4, 15, 12, 5, 3, 10,
	-13, -2, 15, 10, 15, 9, 27, 28,
	9, 3, 8, 11, 24, 20, 14, 4,
	5, -7, 2, 13, 23, 15, 11, 2,
	-6, -8, 3, 14, 9, 17, 3, -1,
	-9, -5, -8, -9, 4, 1, 0, 0,
	-22, -17, -6, 3, -5, -1, 2, -7,
	-25, 11, 8, 14, 9, 14, 2, 94,
	95, 9, -12, 30, 46, 58, 52, 14,
	101, -22, 26, 23, 38, 92, 81, 91,
	5, -16, 1, 9, 27, 28, 38, 38,
	5, -26, -32, -8, -7, 0, -4, 13,
	-25, -31, -24, -12, -17, -7, -4, 30,
	-12, -31, -21, -12, -3, -13, -1, 9,
	-30, -13, -11, -6, -3, 0, 1, -18,
	-9, 20, 25, 24, 26, 24, 27, 28,
	25, 27, 29, 33, 34, 26, 19, 27,
	21, 27, 23, 29, 25, 21, 26, 23,
	24, 23, 28, 27, 27, 19, 27, 21,
	24, 12, 16, 19, 15, 12, 13, 13,
	6, -3, -1, 1, 1, 0, 1, -3,
	-13, -13, -10, -3, -7, -5, -10, -13,
	-23, -11, -11, -3, -6, -8, -8, -5,
	-18, -25, -25, -50, -35, -46, 66, 52,
	94, -27, -32, -4, -28, -45, -5, -13,
	45, -9, 2, -2, 4, -2, 78, 12,
	24, -14, -13, -9, -13, -16, -4, -14,
	-5, -10, -10, -6, -13, -8, -7, 1,
	4, -6, -3, 1, -7, -4, 2, 9,
	2, -9, -2, 1, 4, 5, 12, 15,
	12, -12, -8, -6, 1, -4, -1, -16,
	12, 6, 23, 31, 36, 43, 33, 39,
	34, 11, 27, 43, 55, 59, 31, 58,
	41, 10, 17, 41, 45, 49, 48, 38,
	36, 11, 28, 36, 44, 51, 43, 48,
	33, 9, 10, 20, 37, 33, 26, 32,
	24, -13, 3, 3, 4, 14, 16, 8,
	5, -29, -19, -13, -11, -5, -31, -41,
	-45, -27, -25, -23, -13, -20, -49, -66,
	-27, -186, 45, 23, -271, -25, -101, -8,
	-123, 67, 32, -8, 5, 5, 0, -201,
	59, 45, 17, -109, -147, -134, -165, -183,
	12, 42, -165, -97, -186, -201, -219, -206,
	-255, -24, -34, -76, -213, -168, -146, -164,
	-224, -100, -47, -98, -109, -110, -103, -77,
	-62, -19, -57, -56, -89, -74, -62, -14,
	-13, -19, 14, 5, -45, 2, -58, 20,
	20, -138, -3, 2, 8, 1, 18, 10,
	-124, -25, 42, 42, 30, 31, 48, 44,
	-29, 21, 43, 50, 42, 43, 64, 58,
	27, 4, 36, 46, 46, 48, 52, 43,
	8, -19, 12, 30, 41, 37, 31, 17,
	-6, -23, 2, 17, 25, 26, 17, 3,
	-16, -20, -8, 2, 8, 8, 4, -5,
	-21, -44, -23, -11, -19, -33, -18, -23,
	-48
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

void freeFeatures()
{
	std::vector<Feature> nfeatures;
	std::vector<i16> nfeatureVector;

	features.swap(nfeatures);
	featureVector.swap(nfeatureVector);
}

void extractFeatures()
{
	addFeature("kingCheckPotential", kingCheckPotential, 28);

	ADD_SINGLE_FEATURE(progressBasePly);
	ADD_SINGLE_FEATURE(progressScale);

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
