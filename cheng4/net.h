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

#pragma once

#include <vector>
#include "types.h"
#include "platform.h"

namespace cheng4
{

enum Topology
{
	topo0 = 736,
	topo1 = 448,
	// new: x2 (perspective)
	topo1in = topo1*2,
	topo2 = 1,

	topoLayers = 2
};

typedef i32 fixedp;
typedef i16 wfixedp;

// layer 1 bitplane fast update cache
struct NetCache
{
	// actual cache for layer 1 output, including biases
	wfixedp cache[topo1];
};

struct NetLayerBase
{
	// weights now transposed
	wfixedp *weights = nullptr;
	wfixedp *bias = nullptr;

	virtual ~NetLayerBase() {}

	virtual void init(wfixedp *wvec, wfixedp *bvec)=0;
	virtual void transpose_weights() = 0;

	virtual void cache_init(const i32 *inputIndex, int indexCount, NetCache &cache)=0;

	virtual int getInputSize() const = 0;
	virtual int getOutputSize() const = 0;
};

#define NET_TRANSPOSE_LAYER0_ONLY 1

static constexpr int fixedp_shift = 9;

// we can do with 32-bit mult result because abs(weights) should never exceed 1 << fixedp_shift, ditto for biases
typedef int32_t fixedp_result;

inline fixedp fixed_mul(fixedp a, fixedp b)
{
    return fixedp((fixedp_result)a * b >> fixedp_shift);
}

template<int inputSize, int outputSize, bool last>
struct NetLayer : NetLayerBase
{
	// 32767/(64+1) = 504 => cache: bias + 64 squares = 65
	static constexpr int fixedp_max = 504;

	void init(wfixedp *wvec, wfixedp *bvec) override
	{
		weights = wvec;
		bias = bvec;

		// note: for inference we don't need random init here
	}

	void transpose_weights() override
	{
		transpose_weights_internal(weights);
	}

	void transpose_weights_internal(wfixedp *wptr)
	{
		int w = getInputSize();
		int h = getOutputSize();

		if (w <= 1 || h <= 1)
			return;

		std::vector<wfixedp> tmp(w*h);

		const wfixedp *fptr = wptr;

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
		return last ? value : (value < 0 ? 0 : value > fixedp_max ? fixedp_max : value);
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
		wfixedp *tmp = cache.cache;

		for (int i=0; i<outputSize; i++)
			tmp[i] = bias[i];

		for (int c=0; c<indexCount; c++)
		{
			int i = inputIndex[c];

			const wfixedp *w = weights + i*outputSize;

			CHENG_AUTO_VECTORIZE_LOOP
			for (int j=0; j<outputSize; j++)
				tmp[j] += w[j];
		}
	}

	void cache_add_index(NetCache & CHENG_PTR_NOALIAS cache, i32 index)
	{
		wfixedp *tmp = cache.cache;
		const wfixedp *w = weights + index*outputSize;

		CHENG_AUTO_VECTORIZE_LOOP
		for (int j=0; j<outputSize; j++)
			tmp[j] += w[j];
	}

	void cache_sub_index(NetCache & CHENG_PTR_NOALIAS cache, i32 index)
	{
		wfixedp *tmp = cache.cache;
		const wfixedp *w = weights + index*outputSize;

		CHENG_AUTO_VECTORIZE_LOOP
		for (int j=0; j<outputSize; j++)
			tmp[j] -= w[j];
	}

	// forward, cached
	void forward_cache(const NetCache & CHENG_PTR_NOALIAS cache, wfixedp * CHENG_PTR_NOALIAS output)
	{
		const wfixedp *tmp = cache.cache;

		CHENG_AUTO_VECTORIZE_LOOP
		for (int i=0; i<outputSize; i++)
			output[i] = activate(tmp[i]);
	}

	// feedforward
	void forward(const wfixedp *  CHENG_PTR_NOALIAS input, fixedp * CHENG_PTR_NOALIAS output)
	{
		fixedp_result tmp[outputSize];

		CHENG_AUTO_VECTORIZE_LOOP
		for (int i=0; i<outputSize; i++)
			tmp[i] = (fixedp_result)bias[i] << fixedp_shift;

#if NET_TRANSPOSE_LAYER0_ONLY
		for (int i=0; i<outputSize; i++)
		{
			const wfixedp *w = weights + i*inputSize;

			fixedp_result tmpdot = 0;

			CHENG_AUTO_VECTORIZE_LOOP
			for (int j=0; j<inputSize; j++)
				tmpdot += (fixedp_result)input[j] * w[j];

			tmp[i] += tmpdot;
		}
#else
		for (int i=0; i<inputSize; i++)
		{
			const wfixedp *w = weights + i*outputSize;

			fixedp_result inputw = input[i];

			// note: we bet we don't overflow here - that weights are relatively small
			// note2: preshift by 8 did hurt the output waay to much to be usable
			// this is much much slower than float so I'll probably have to go with only 1 hidden layer
			CHENG_AUTO_VECTORIZE_LOOP
			for (int j=0; j<outputSize; j++)
				tmp[j] += inputw * w[j];
		}
#endif

		CHENG_AUTO_VECTORIZE_LOOP
		for (int i=0; i<outputSize; i++)
			output[i] = activate((fixedp)(tmp[i] >> fixedp_shift));
	}
};

struct Network
{
	std::vector<NetLayerBase *> layers;

	// now fixed
	NetLayer<topo0, topo1, false> layer0;
	NetLayer<topo1in, 1, true> layer1;

	// all weights, including biases (biases come at the end)
	std::vector<wfixedp> weights;
	// this is where aligned weights start
	int weight_index;
	// total number of weights
	int weight_size;
	// this is where biases start in weights
	int bias_index;

	bool load(const char *filename);

	bool load_buffer(const void *buf, int size);

	bool load_buffer_compressed(const void *buf, int size);

	bool init_topology();

	void forward_cache(const NetCache &cache, const NetCache &cacheOpp, fixedp *outp, int outpsize);

	void cache_init(const i32 *nonzero, int nzcount, NetCache &cache);

	void cache_add_index(NetCache &cache, i32 index);
	void cache_sub_index(NetCache &cache, i32 index);

	void transpose_weights();

	static int32_t to_centipawns(fixedp w);
};

}
