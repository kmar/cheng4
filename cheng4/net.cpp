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
#include <vector>
#include <fstream>

//#include <intrin.h>

namespace cheng4
{

#if defined(__clang__)
#	define AUTO_VECTORIZE_LOOP _Pragma("clang loop vectorize(enable)")
#else
#	define AUTO_VECTORIZE_LOOP
#endif


static constexpr int MAX_LAYER_SIZE = 768;

template<int inputSize, int outputSize, bool last>
struct NetLayer : NetLayerBase
{
	void init(float *wvec, float *bvec) override
	{
		weights = wvec;
		bias = bvec;

		// note: for inference we don't need random init here
	}

	void transpose_weights() override
	{
		transpose_weights_internal(weights);
	}

	void transpose_biases() override
	{
		transpose_weights_internal(bias);
	}

	void transpose_weights_internal(float *wptr)
	{
		int w = getInputSize();
		int h = getOutputSize();

		if (w <= 1 || h <= 1)
			return;

		std::vector<float> tmp(w*h);

		const float *fptr = wptr;

		//printf("wcount=%d\n", w*h);
		for (int y=0; y<h; y++)
			for (int x=0; x<w; x++)
				tmp[x*h+y] = *fptr++;

		for (int i=0; i<w*h; i++)
			wptr[i] = tmp[i];
	}

	// leaky relu/copy
	static inline float activate(float value)
	{
		return last ? value : (value < 0.0f ? value * 0.01f : value);
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
		float *tmp = cache.cache;

		memcpy(tmp, bias, outputSize*sizeof(float));

		for (int c=0; c<indexCount; c++)
		{
			int i = inputIndex[c];

			const float *w = weights + i*outputSize;

			AUTO_VECTORIZE_LOOP
			for (int j=0; j<outputSize; j++)
				tmp[j] += w[j];
		}
	}

	void cache_add_index(NetCache &cache, i32 index) override
	{
		float *tmp = cache.cache;
		const float *w = weights + index*outputSize;

		AUTO_VECTORIZE_LOOP
		for (int j=0; j<outputSize; j++)
			tmp[j] += w[j];
	}

	void cache_sub_index(NetCache &cache, i32 index) override
	{
		float *tmp = cache.cache;
		const float *w = weights + index*outputSize;

		AUTO_VECTORIZE_LOOP
		for (int j=0; j<outputSize; j++)
			tmp[j] -= w[j];
	}

	// forward, cached
	void forward_cache(const NetCache &cache, float *output) override
	{
		const float *tmp = cache.cache;

		AUTO_VECTORIZE_LOOP
		for (int i=0; i<outputSize; i++)
			output[i] = activate(tmp[i]);
	}

	// restricted feedforward with many zero weights
	// inputIndex = indices with non-zero weights (=1.0)
	void forward_restricted(const i32 *inputIndex, int indexCount, float *output) override
	{
		float tmp[MAX_LAYER_SIZE];

		memcpy(tmp, bias, outputSize*sizeof(float));

		for (int c=0; c<indexCount; c++)
		{
			int i = inputIndex[c];

			const float *w = weights + i*outputSize;

			AUTO_VECTORIZE_LOOP
			for (int j=0; j<outputSize; j++)
				tmp[j] += w[j];
		}

		AUTO_VECTORIZE_LOOP
		for (int i=0; i<outputSize; i++)
			output[i] = activate(tmp[i]);
	}

	// feedforward
	void forward(const float *input, float *output) override
	{
		float tmp[MAX_LAYER_SIZE];

		memcpy(tmp, bias, outputSize*sizeof(float));

		for (int i=0; i<inputSize; i++)
		{
			const float *w = weights + i*outputSize;

			auto inputw = input[i];

			AUTO_VECTORIZE_LOOP
			for (int j=0; j<outputSize; j++)
				tmp[j] += inputw * w[j];
		}

		AUTO_VECTORIZE_LOOP
		for (int i=0; i<outputSize; i++)
			output[i] = activate(tmp[i]);
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

void Network::forward_nz(const float *inp, int inpsize, const i32 *nonzero, int nzcount, float *outp, int outpsize)
{
	assert(inpsize >= layers[0]->getInputSize());
	assert(outpsize >= layers[layers.size()-1]->getOutputSize());
	(void)inpsize;
	(void)outpsize;
	assert(inpsize >= layers[0]->getInputSize());
	assert(outpsize >= layers[layers.size()-1]->getOutputSize());

	float temp[MAX_LAYER_SIZE];
	float temp2[MAX_LAYER_SIZE];

	for (int i=0; i<(int)layers.size(); i++)
	{
		if (i == 0)
			layers[i]->forward_restricted(nonzero, nzcount, temp);
		else
			layers[i]->forward(inp, temp);

		auto osz = layers[i]->getOutputSize();

		AUTO_VECTORIZE_LOOP
		for (int j=0; j<osz; j++)
			temp2[j] = temp[j];

		inp = temp2;
	}

	int osz = layers[layers.size()-1]->getOutputSize();

	AUTO_VECTORIZE_LOOP
	for (int j=0; j<osz; j++)
		outp[j] = temp[j];
}

void Network::forward_cache(const NetCache &cache, float *outp, int outpsize)
{
	assert(outpsize >= layers[layers.size()-1]->getOutputSize());
	(void)outpsize;
	assert(outpsize >= layers[layers.size()-1]->getOutputSize());

	float temp[MAX_LAYER_SIZE];
	float temp2[MAX_LAYER_SIZE];

	const float *inp = nullptr;

	for (int i=0; i<(int)layers.size(); i++)
	{
		if (i == 0)
			layers[i]->forward_cache(cache, temp);
		else
			layers[i]->forward(inp, temp);

		auto osz = layers[i]->getOutputSize();

		AUTO_VECTORIZE_LOOP
		for (int j=0; j<osz; j++)
			temp2[j] = temp[j];

		inp = temp2;
	}

	int osz = layers[layers.size()-1]->getOutputSize();

	AUTO_VECTORIZE_LOOP
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

void Network::transpose_biases()
{
	for (auto &it : layers)
		it->transpose_biases();
}

bool Network::load(const char *filename)
{
	std::ifstream ifs(filename, std::ios::in | std::ios::binary);

	ifs.read((char *)(weights.data() + weight_index), weight_size * sizeof(float));

	return !ifs.fail();
}

bool Network::load_buffer(const void *ptr, int size)
{
	if (size != weight_size * (int)sizeof(float))
		return false;

	memcpy(weights.data() + weight_index, ptr, weight_size * sizeof(float));
	return true;
}

float Network::to_centipawns(float w)
{
	// convert win prob back to score
	w = w < 0.0f ? 0.0f : w > 1.0f ? 1.0f : w;

	// don't get overexcited with insane evals very close to 0 or 1
	w -= 0.5f;
	w *= 0.999999f;
	w += 0.5f;

	// last cheng HCE K for texel tuning
	constexpr float HCE_K = 1.25098f;

	float res = (-173.718f/HCE_K) * logf(1.0f / w - 1.0f);
	return res;
}

}
