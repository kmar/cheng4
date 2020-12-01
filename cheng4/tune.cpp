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
	-45, -1, -16, -25, -21, -8, 1,
	16, 32, 92, 176, 260, 355, 497, 643,
	595, 1007, 1088, 1641, 1838, 1570, 2985, 1842,
	2622, 2483, 2247, 5, 252, 0, -1, 121,
	25, -52, 72, 156, 409, 741, 34, -50,
	58, 66, 156, 320, 402, 56, 187, 222,
	7, -26, 101, 98, 152, 57, 316, 311,
	397, 991, 92, 342, 340, 552, 1024, 12,
	12, 5, 4, 7, -10, 19, 11, 15,
	14, 192, 140, 326, 567, 1023, 872, 2646,
	-8, 65, 25, 105, 407, 365, 361, 532,
	499, 331, 194, 4, 418, 126, 2, 48,
	88, 24, 129, 186, 218, 170, 118, 80,
	0, 0, -129, 55, 110, 34, 82, 202,
	1106, 244, 116, -7, -4, 77, 210, 556,
	0, -2, 64, 168, 357, 992, 0, 15,
	-129, -123, -8, 298, 947, 173, 159, 427,
	733, 1271, 1847, -480, -311, -242, -181, -131,
	-94, -38, 23, 126, -462, -519, -264, -156,
	-56, 51, 55, 55, -29, -261, -157, -63,
	-16, 37, 61, 79, 53, 52, 67, 95,
	125, 292, 622, -311, -222, -127, -33, 77,
	147, 195, 222, 233, 223, 193, 213, 167,
	242, 37, 68, 122, 82, 64, 113, 146,
	208, 251, 330, 382, 381, 375, 575, -74,
	-331, -4, 116, 253, 309, 354, 402, 440,
	494, 527, 561, 576, 583, 501, 636, 114,
	217, 377, 379, 389, 375, 380, 371, 363,
	364, 386, 402, 410, 525, 589, 786, 1037,
	1556, 1733, 1878, 1410, 1909, 927, 691, -934,
	-3949, 3916, 1168, -65, 200, 267, 462, 548,
	631, 679, 774, 850, 918, 948, 962, 986,
	934, 952, 914, 873, 795, 813, 802, 843,
	872, 823, 923, 895, 1094, 846, 1249, 947,
	-108, 44, -22, -59, 40, 71, 81, 35,
	79, 149, 137, 173, 154, 292, 272, 661,
	-2085, -143, -102, -193, -57, -63, -20, -5,
	172, 70, 106, 152, 171, 189, 253, 455,
	-132, 114, 16, 57, 96, 78, -19, -144,
	-95, 6, 3, 33, 31, 50, 72, 31,
	15, 2, 7, 8, 18, 24, 29, 4,
	8, -4, 1, 8, 12, 17, 19, 10,
	8, -9, -4, -4, -4, 9, -5, 15,
	0, -11, -7, -6, -10, -4, 11, 25,
	-9, 52, 58, 50, 18, 7, 17, 54,
	36, 43, 37, 14, -10, -21, 3, 13,
	13, 27, 15, 0, -17, -14, -8, 1,
	1, 14, 8, -5, -11, -9, -4, -2,
	-8, 10, 4, -4, -5, 0, -1, -2,
	-7, 12, 6, -1, 5, 5, 8, 0,
	-11, -187, -119, -50, -20, 36, -116, -66,
	-171, -14, -17, 0, 28, -16, 51, -31,
	-6, -24, -2, 8, 21, 81, 53, 25,
	-33, -9, 7, 17, 28, 5, 36, 9,
	23, 3, 18, 12, 9, 12, 15, 4,
	6, -17, -3, 2, 10, 17, 8, 13,
	-9, 2, -6, -9, 6, 6, 6, -5,
	4, -52, -16, -7, -1, 0, 9, -9,
	-53, -4, -2, 1, 17, 5, -5, -4,
	-54, -2, 9, 6, 14, 12, 2, 0,
	-6, 1, 6, 20, 24, 12, 29, 7,
	2, 11, 7, 18, 22, 19, 22, 15,
	11, 8, 4, 20, 19, 28, 16, 13,
	10, -19, -2, -1, 16, 16, 5, 4,
	1, -39, -8, -14, 2, 5, -1, 5,
	-1, -56, -11, -12, -2, 0, -6, 1,
	-23, -43, -44, -93, -62, -91, -119, -19,
	29, -41, -24, -27, -49, -42, -75, -77,
	-67, -12, -4, -13, 9, -2, 52, -8,
	7, -23, -7, 1, 20, 5, -10, -4,
	-29, -25, 1, -8, 10, -2, -9, -10,
	1, -11, -3, 1, -6, -3, 0, 6,
	1, 0, -1, 1, -9, 1, 6, 19,
	14, 2, 19, -13, -11, 10, -12, 7,
	-4, 6, 13, 6, 17, 6, -4, 5,
	-4, 16, 7, 7, 3, 6, 6, 10,
	-9, -2, 15, 10, 7, 13, 28, 23,
	1, 4, 11, 15, 24, 19, 16, 8,
	6, -9, 2, 12, 23, 18, 7, 5,
	-9, -5, 4, 13, 5, 19, 5, 0,
	-7, -6, -4, -5, 2, -3, 3, 0,
	-30, -6, -9, 1, -3, -5, -1, -13,
	-15, -1, 7, -7, 10, -19, 1, 45,
	95, 5, -13, 17, 29, 30, 52, 49,
	78, -12, 17, -5, 17, 61, 64, 75,
	34, -17, -15, 10, 30, 22, 19, 21,
	28, -32, -25, -8, -4, -12, -10, 4,
	-23, -27, -13, -8, -17, 0, -16, 6,
	-20, -32, -21, -12, -5, -11, -6, 4,
	-31, -12, -8, -6, 0, 0, -3, 3,
	-8, 14, 21, 20, 21, 23, 23, 21,
	20, 27, 29, 32, 33, 31, 21, 25,
	24, 27, 24, 28, 27, 25, 24, 21,
	24, 25, 29, 28, 26, 22, 25, 23,
	18, 11, 18, 18, 13, 12, 16, 13,
	5, -4, 0, -3, 1, -2, 1, 1,
	-13, -18, -7, -2, -7, -5, -9, -8,
	-16, -12, -11, -5, -6, -10, -8, -9,
	-17, -26, -26, -54, -54, -47, -47, 98,
	45, -16, -29, -1, -13, -29, -9, -5,
	43, -13, -11, -19, -11, -15, 68, 37,
	20, -9, -5, -9, -14, -6, -5, 0,
	-2, -9, -9, -11, -9, -13, -9, 0,
	-17, -9, -1, -3, -11, -6, -2, 9,
	4, -15, -6, 2, 5, 3, 7, 28,
	-8, -9, -8, 4, 1, -12, 4, -5,
	-5, 10, 20, 34, 36, 34, 36, 20,
	30, 14, 27, 40, 48, 47, 31, 26,
	42, 10, 31, 39, 47, 54, 38, 29,
	29, 10, 26, 38, 44, 44, 37, 44,
	33, 4, 14, 21, 33, 30, 24, 28,
	20, -7, -3, 7, 4, 17, 12, 13,
	1, -18, -10, -12, -12, -5, -30, -44,
	-14, -23, -23, -20, -18, -12, -51, -47,
	-24, -395, 36, 23, -326, -26, -103, 18,
	84, 67, 31, -7, 5, -95, 0, 2,
	26, 46, 20, -107, 60, -135, -174, -179,
	-2, 48, 17, -68, -185, -178, -92, -127,
	-205, -18, -36, -77, -213, -168, -100, -163,
	-132, -132, 16, -64, -110, -63, -104, -76,
	-66, -16, -89, -45, -94, -65, -54, -13,
	-12, -37, -10, 9, -61, 3, -53, 20,
	20, -153, -32, -18, 1, 2, 18, 21,
	-110, -44, 25, 26, 22, 42, 49, 64,
	-33, 9, 30, 40, 45, 40, 62, 57,
	26, 0, 18, 36, 41, 42, 45, 41,
	13, -19, 11, 24, 35, 34, 29, 19,
	-6, -29, -4, 10, 21, 23, 16, 4,
	-10, -16, -9, 3, 3, 4, 2, -5,
	-18, -46, -18, -11, -20, -35, -19, -23,
	-44
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
