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

#define MLZ_DEC_MINI_IMPLEMENTATION
#include "nets/mlz/mlz_dec_mini.h"

namespace cheng4
{

inline fixedp fixed_mul(fixedp a, fixedp b)
{
    return (int64_t)a * b >> 16;
}

static constexpr int MAX_LAYER_SIZE = 768;

template<int inputSize, int outputSize, bool last>
struct NetLayer : NetLayerBase
{
	void init(fixedp *wvec, fixedp *bvec) override
	{
		weights = wvec;
		bias = bvec;

		// note: for inference we don't need random init here
	}

	void transpose_weights() override
	{
		transpose_weights_internal(weights);
	}

	void transpose_weights_internal(fixedp *wptr)
	{
		int w = getInputSize();
		int h = getOutputSize();

		if (w <= 1 || h <= 1)
			return;

		std::vector<fixedp> tmp(w*h);

		const fixedp *fptr = wptr;

		//printf("wcount=%d\n", w*h);
		for (int y=0; y<h; y++)
			for (int x=0; x<w; x++)
				tmp[x*h+y] = *fptr++;

		for (int i=0; i<w*h; i++)
			wptr[i] = tmp[i];
	}

	// relu/copy
	static inline fixedp activate(fixedp value)
	{
		return last ? value : (value < 0 ? 0 : value);
	}

	int getInputSize() const override
	{
		return inputSize;
	}

	int getOutputSize() const override
	{
		return outputSize;
	}

	void cache_init(const i32 *inputIndex, int indexCount, NetCache &cache) override
	{
		fixedp *tmp = cache.cache;

		memcpy(tmp, bias, outputSize*sizeof(fixedp));

		for (int c=0; c<indexCount; c++)
		{
			int i = inputIndex[c];

			const fixedp *w = weights + i*outputSize;

			CHENG_AUTO_VECTORIZE_LOOP
			for (int j=0; j<outputSize; j++)
				tmp[j] += w[j];
		}
	}

	void cache_add_index(NetCache & CHENG_PTR_NOALIAS cache, i32 index) override
	{
		fixedp *tmp = cache.cache;
		const fixedp *w = weights + index*outputSize;

		CHENG_AUTO_VECTORIZE_LOOP
		for (int j=0; j<outputSize; j++)
			tmp[j] += w[j];
	}

	void cache_sub_index(NetCache & CHENG_PTR_NOALIAS cache, i32 index) override
	{
		fixedp *tmp = cache.cache;
		const fixedp *w = weights + index*outputSize;

		CHENG_AUTO_VECTORIZE_LOOP
		for (int j=0; j<outputSize; j++)
			tmp[j] -= w[j];
	}

	// forward, cached
	void forward_cache(const NetCache & CHENG_PTR_NOALIAS cache, fixedp * CHENG_PTR_NOALIAS output) override
	{
		const fixedp *tmp = cache.cache;

		CHENG_AUTO_VECTORIZE_LOOP
		for (int i=0; i<outputSize; i++)
			output[i] = activate(tmp[i]);
	}

	// restricted feedforward with many zero weights
	// inputIndex = indices with non-zero weights (=1.0)
	void forward_restricted(const i32 * CHENG_PTR_NOALIAS inputIndex, int indexCount, fixedp * CHENG_PTR_NOALIAS output) override
	{
		fixedp tmp[outputSize];

		memcpy(tmp, bias, outputSize*sizeof(fixedp));

		for (int c=0; c<indexCount; c++)
		{
			int i = inputIndex[c];

			const fixedp *w = weights + i*outputSize;

			CHENG_AUTO_VECTORIZE_LOOP
			for (int j=0; j<outputSize; j++)
				tmp[j] += w[j];
		}

		CHENG_AUTO_VECTORIZE_LOOP
		for (int i=0; i<outputSize; i++)
			output[i] = activate(tmp[i]);
	}

	// feedforward
	void forward(const fixedp *  CHENG_PTR_NOALIAS input, fixedp * CHENG_PTR_NOALIAS output) override
	{
		int64_t tmp[outputSize];

		CHENG_AUTO_VECTORIZE_LOOP
		for (int i=0; i<outputSize; i++)
			tmp[i] = (int64_t)bias[i] << 16;

		for (int i=0; i<inputSize; i++)
		{
			const fixedp *w = weights + i*outputSize;

			int64_t inputw = input[i];

			// note: we bet we don't overflow here - that weights are relatively small
			// note2: preshift by 8 did hurt the output waay to much to be usable
			// this is much much slower than float so I'll probably have to go with only 1 hidden layer
			CHENG_AUTO_VECTORIZE_LOOP
			for (int j=0; j<outputSize; j++)
				tmp[j] += inputw * w[j];
		}

		CHENG_AUTO_VECTORIZE_LOOP
		for (int i=0; i<outputSize; i++)
			output[i] = activate((fixedp)(tmp[i] >> 16));
	}
};

Network::~Network()
{
	for (auto *it : layers)
		delete it;
}

void Network::cache_init(const i32 *nonzero, int nzcount, NetCache &cache)
{
	layers[0]->cache_init(nonzero, nzcount, cache);
}

void Network::forward_nz(const fixedp * CHENG_PTR_NOALIAS inp, int inpsize, const i32 * CHENG_PTR_NOALIAS nonzero, int nzcount, fixedp * CHENG_PTR_NOALIAS outp, int outpsize)
{
	assert(inpsize >= layers[0]->getInputSize());
	assert(outpsize >= layers[layers.size()-1]->getOutputSize());
	(void)inpsize;
	(void)outpsize;
	assert(inpsize >= layers[0]->getInputSize());
	assert(outpsize >= layers[layers.size()-1]->getOutputSize());

	fixedp temp[MAX_LAYER_SIZE];
	fixedp temp2[MAX_LAYER_SIZE];

	for (int i=0; i<(int)layers.size(); i++)
	{
		if (i == 0)
			layers[i]->forward_restricted(nonzero, nzcount, temp);
		else
			layers[i]->forward(inp, temp);

		auto osz = layers[i]->getOutputSize();

		CHENG_AUTO_VECTORIZE_LOOP
		for (int j=0; j<osz; j++)
			temp2[j] = temp[j];

		inp = temp2;
	}

	int osz = layers[layers.size()-1]->getOutputSize();

	CHENG_AUTO_VECTORIZE_LOOP
	for (int j=0; j<osz; j++)
		outp[j] = temp[j];
}

void Network::forward_cache(const NetCache &cache, fixedp * CHENG_PTR_NOALIAS outp, int outpsize)
{
	assert(outpsize >= layers[layers.size()-1]->getOutputSize());
	(void)outpsize;
	assert(outpsize >= layers[layers.size()-1]->getOutputSize());

	fixedp temp[MAX_LAYER_SIZE];
	fixedp temp2[MAX_LAYER_SIZE];

	const fixedp *inp = nullptr;

	for (int i=0; i<(int)layers.size(); i++)
	{
		if (i == 0)
			layers[i]->forward_cache(cache, temp);
		else
			layers[i]->forward(inp, temp);

		auto osz = layers[i]->getOutputSize();

		CHENG_AUTO_VECTORIZE_LOOP
		for (int j=0; j<osz; j++)
			temp2[j] = temp[j];

		inp = temp2;
	}

	int osz = layers[layers.size()-1]->getOutputSize();

	CHENG_AUTO_VECTORIZE_LOOP
	for (int j=0; j<osz; j++)
		outp[j] = temp[j];
}

void Network::cache_add_index(NetCache &cache, i32 index)
{
	layers[0]->cache_add_index(cache, index);
}

void Network::cache_sub_index(NetCache &cache, i32 index)
{
	layers[0]->cache_sub_index(cache, index);
}

struct LayerDesc
{
	int insize;
	int outsize;
	bool last;
	NetLayerBase *(*create_func)();
};

#define NET_FIXED_LAYER_DESC(insz, outsz, last) {insz, outsz, last, []()->NetLayerBase *{return new NetLayer<insz, outsz, last>;}}

// we only support these fixed layer topologies
static LayerDesc layerDesc[] =
{
	NET_FIXED_LAYER_DESC(topo0, topo1, false),
	NET_FIXED_LAYER_DESC(topo1, topo2, false),
	NET_FIXED_LAYER_DESC(topo2, 1, true)
};

static LayerDesc layerDesc2[] =
{
	NET_FIXED_LAYER_DESC(topo0, topo1, false),
	NET_FIXED_LAYER_DESC(topo1, 1, true)
};

#undef NET_FIXED_LAYER_DESC

bool Network::init_topology(const int *sizes, int count)
{
	assert(count>1);
	layers.resize(count-1);

	int total = 0;
	int total_nobias = 0;

	for (int i=0; i<count-1; i++)
	{
		total += (sizes[i]+1) * sizes[i+1];
		total_nobias += sizes[i] * sizes[i+1];
	}

	weight_size = total;

	//"total=%d nobias=%d\n", total, total_nobias;
	bias_index = total_nobias;

	weights.resize(total + 16);

	// align to cacheline
	auto ptr = (uintptr_t)weights.data();
	auto aptr = ptr;
	aptr = (aptr + 63) & ~(uintptr_t)63;

	weight_index = int((aptr - ptr)/4);

	//printf("aligned_weight_index = %d\n", weight_index);

	bias_index += weight_index;

	int widx = weight_index;
	int bidx = bias_index;

	const int numLayouts = topoLayers;

	const auto *layerDescPtr = topoLayers > 2 ? layerDesc : layerDesc2;

	for (int i=0; i<count-1; i++)
	{
		const bool islast = i+1 == count-1;

		NetLayerBase *nlayer = nullptr;

		for (int j=0; j<numLayouts; j++)
		{
			const LayerDesc *l = &layerDescPtr[j];

			if (sizes[i] == l->insize && sizes[i+1] == l->outsize && islast == l->last)
			{
				nlayer = l->create_func();
				break;
			}
		}

		if (!nlayer)
			return false;

		layers[i] = nlayer;

		layers[i]->init(weights.data() + widx, weights.data() + bidx);

		widx += sizes[i] * sizes[i+1];
		bidx += sizes[i+1];
	}

	return true;
}

void Network::transpose_weights()
{
	for (auto &it : layers)
		it->transpose_weights();
}

bool Network::load(const char *filename)
{
	std::ifstream ifs(filename, std::ios::in | std::ios::binary);

	ifs.read((char *)(weights.data() + weight_index), weight_size * sizeof(fixedp));

	return !ifs.fail();
}

bool Network::load_buffer(const void *ptr, int size)
{
	if (size != weight_size * (int)sizeof(fixedp))
		return false;

	memcpy(weights.data() + weight_index, ptr, weight_size * sizeof(fixedp));

	return true;
}

bool Network::load_buffer_compressed(const void *ptr, int size)
{
	int usize = mlz_decompress_mini(weights.data() + weight_index, ptr, size);

	return usize == weight_size * (int)sizeof(fixedp);
}

int32_t Network::to_centipawns(fixedp w)
{
	return (fixed_mul(w, 100*65536) + 32768) >> 16;
}

}
