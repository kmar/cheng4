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

namespace cheng4
{

struct NetLayerBase
{
	// weights now transposed
	float *weights = nullptr;
	float *bias = nullptr;

	virtual ~NetLayerBase() {}
	virtual void init(float *wvec, float *bvec)=0;
	virtual void transpose_weights() = 0;
	virtual void transpose_biases() = 0;
	virtual void forward_restricted(const i32 *inputIndex, int indexCount, float *output)=0;
	virtual void forward(const float *input, float *output)=0;
	virtual int getInputSize() const = 0;
	virtual int getOutputSize() const = 0;
};

struct Network
{
	std::vector<NetLayerBase *> layers;
	// all weights, including biases (biases come at the end)
	std::vector<float> weights;
	// this is where aligned weights start
	int weight_index;
	// total number of weights
	int weight_size;
	// this is where biases start in weights
	int bias_index;

	~Network();

	bool load(const char *filename);

	bool load_buffer(const void *buf, int size);

	bool init_topology(const int *sizes, int count);

	void forward_nz(const float *inp, int inpsize, i32 *nonzero, int nzcount, float *outp, int outpsize);

	void transpose_weights();

	void transpose_biases();

	static float to_centipawns(float w);
};

}
