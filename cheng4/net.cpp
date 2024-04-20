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

#include "net.h"
#include "types.h"
#include "platform.h"
#include <vector>
#include <fstream>
#include <algorithm>

#define MLZ_DEC_MINI_IMPLEMENTATION
#include "nets/mlz/mlz_dec_mini.h"

namespace cheng4
{

void Network::cache_init(const i32 *nonzero, int nzcount, NetCache &cache)
{
	layers[0]->cache_init(nonzero, nzcount, cache);
}

void Network::forward_cache(const NetCache & CHENG_PTR_NOALIAS cache, const NetCache & CHENG_PTR_NOALIAS cacheOpp, fixedp * CHENG_PTR_NOALIAS outp, int outpsize)
{
	assert(outpsize >= layers[layers.size()-1]->getOutputSize());
	(void)outpsize;
	assert(outpsize >= layers[layers.size()-1]->getOutputSize());

	fixedp temp[MAX_LAYER_SIZE];

	// now manually unpacked
	layer0.forward_cache(cache, temp);
	layer0.forward_cache(cacheOpp, temp + topo1);

	layer1.forward(temp, outp);
}

void Network::cache_add_index(NetCache &cache, i32 index)
{
	layer0.cache_add_index(cache, index);
}

void Network::cache_sub_index(NetCache &cache, i32 index)
{
	layer0.cache_sub_index(cache, index);
}

bool Network::init_topology()
{
	const int numLayouts = topoLayers;
	(void)numLayouts;

	layers.resize(topoLayers);
	layers[0] = &layer0;
	layers[1] = &layer1;

	int total = 0;
	int total_nobias = 0;

	for (int i=0; i<topoLayers; i++)
	{
		int insize = layers[i]->getInputSize();
		int outsize = layers[i]->getOutputSize();
		total += (insize+1) * outsize;
		total_nobias += insize * outsize;
	}

	weight_size = total;

	//"total=%d nobias=%d\n", total, total_nobias;
	bias_index = total_nobias;

	weights.resize(total + 16);

	// align to cacheline
	auto ptr = (uintptr_t)weights.data();
	auto aptr = ptr;
	aptr = (aptr + 63) & ~(uintptr_t)63;

	weight_index = int((aptr - ptr)/sizeof(wfixedp));

	//printf("aligned_weight_index = %d\n", weight_index);

	bias_index += weight_index;

	int widx = weight_index;
	int bidx = bias_index;

	assert(topoLayers == numLayouts);

	for (int i=0; i<topoLayers; i++)
	{
		auto *nlayer = layers[i];

		nlayer->init(weights.data() + widx, weights.data() + bidx);

		widx += nlayer->getInputSize() * nlayer->getOutputSize();
		bidx += nlayer->getOutputSize();
	}

	return true;
}

void Network::transpose_weights()
{
	for (size_t i=0; i<layers.size();
#if !NET_TRANSPOSE_LAYER0_ONLY
		i++
#endif
	)
	{
		auto &it = layers[i];
		it->transpose_weights();
#if NET_TRANSPOSE_LAYER0_ONLY
		break;
#endif
	}
}

bool Network::load(const char *filename)
{
	std::ifstream ifs(filename, std::ios::in | std::ios::binary);

	ifs.read((char *)(weights.data() + weight_index), weight_size * sizeof(wfixedp));

	return !ifs.fail();
}

bool Network::load_buffer(const void *ptr, int size)
{
	if (size != weight_size * (int)sizeof(wfixedp))
		return false;

	memcpy(weights.data() + weight_index, ptr, weight_size * sizeof(wfixedp));

	return true;
}

bool Network::load_buffer_compressed(const void *ptr, int size)
{
	int usize = mlz_decompress_mini(weights.data() + weight_index, ptr, size);

	return usize == weight_size * (int)sizeof(wfixedp);
}

int32_t Network::to_centipawns(fixedp w)
{
	return (fixed_mul(w, 100*(1 << fixedp_shift)) + ((1 << fixedp_shift)-1)) >> fixedp_shift;
}

}
