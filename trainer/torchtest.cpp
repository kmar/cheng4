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

#include "pch.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "memmap.h"

#include "../cheng4/net.h"
#include "net_indices.h"

constexpr int PACKED_TRAIN_ENTRY_SIZE = 38;

// as big as we can fit into memory
constexpr int BATCH_SIZE = 1024*1024*1;
constexpr int INPUT_SIZE = cheng4::topo0;

// 1% per epoch
constexpr double EPOCH_LR_DECAY_RATE = 0.99;

constexpr char NET_FILENAME[] = "test.net";
constexpr char NET_FP_FILENAME[] = "test.fpnet";

// last cheng HCE K for texel tuning
constexpr double HCE_K = 1.25098;

// profile batch time?
constexpr bool profile = false;

// convert eval score (cp) to win prob
template<typename T>
T sigmoid(T s)
{
	return 1.0 / (1.0 + pow(10.0, -HCE_K*s/400.0));
}

// convert win prob back to score (cp)
double inverse_sigmoid(double w)
{
	w = std::clamp(w, 0.0, 1.0);

	double res = -173.718 * log(1.0 / w - 1.0) / HCE_K;
	return std::clamp(res, -12800.0, 12800.0);
}

struct labeled_position
{
	float label;
	int16_t score;
	int16_t outcome;
	int16_t flags;
	uint8_t pieces[32];

	labeled_position() = default;

	labeled_position(const labeled_position &) = default;
	labeled_position(labeled_position &&) = default;

	labeled_position &operator =(const labeled_position &) = default;
	labeled_position &operator =(labeled_position &&) = default;
};

float label_position(const labeled_position &p)
{
	float res;
	// label: pawn score (1.0 = pawn)
	res = p.label;
	// assume 20.0 pawns = win
	float outcome = ((float)p.outcome - 1.0f) * 20.0f;
	// mix search and outcome, 50%
	res = res * 0.5f + outcome * 0.5f;

	return res;
}

void unpack_position_fast(void *dstp, void *dstp_opp, const labeled_position &pos)
{
	auto *dst = static_cast<float *>(dstp);
	auto *dst_opp = static_cast<float *>(dstp_opp);

	int16_t ninds[64];
	bool blackToMove = (pos.flags & 1) != 0;
	auto count = (int16_t)netIndices(blackToMove, pos.pieces, ninds);

	for (int i=0; i<count; i++)
	{
		dst[ninds[i]] = 1.0f;
		dst_opp[flipNetIndex(ninds[i])] = 1.0f;
	}
}

void unpack_position(void *dstp, void *dstp_opp, const labeled_position &pos)
{
	auto *dst = static_cast<float *>(dstp);
	auto *dst_opp = static_cast<float *>(dstp_opp);

	for (int i=0; i<INPUT_SIZE; i++)
		dst[i] = dst_opp[i] = 0.0f;

	unpack_position_fast(dst, dst_opp, pos);
}

static size_t tensor_size(torch::Tensor t)
{
	size_t res = 1;

	for (auto sz : t.sizes())
		res *= (size_t)sz;

	return res;
}

std::vector<float> unpack_tensor(torch::Tensor t)
{
	std::vector<float> res;

	size_t tsize = tensor_size(t);

	res.resize(tsize);

	auto *ptr = static_cast<const float *>(t.data_ptr());

	for (size_t i=0; i<tsize; i++)
		res[i] = ptr[i];

	return res;
}

void pack_tensor(torch::Tensor t, const float *src)
{
	size_t tsize = tensor_size(t);

	auto *dst = static_cast<float *>(t.mutable_data_ptr());

	for (size_t i=0; i<tsize; i++)
		dst[i] = src[i];
}

labeled_position mem_load_position(const uint8_t *&ptr, const uint8_t *end)
{
#define mem_read_int(var) \
	if (ptr + sizeof(var) > end) \
		break; \
	memcpy(&(var), ptr, sizeof(var)); \
	ptr += sizeof(var)

#define mem_read_buf(var, sz) \
	if (ptr + sz > end) \
		break; \
	memcpy((var), ptr, sz); \
	ptr += sz

	int16_t tmp;

	labeled_position p;

	do
	{
		mem_read_int(tmp);

		p.score = tmp;
		p.label = tmp/100.0f;

		mem_read_int(p.outcome);
		mem_read_int(p.flags);

		bool blackToMove = (p.flags & 1) != 0;

		// also: indices are from stm's point of view!
		if (blackToMove)
			p.outcome = 2 - p.outcome;

		// now using nibble-packed indices
		mem_read_buf(p.pieces, 32);
	} while(false);

	return p;
}

memory_mapped_file load_trainfile(const char *fn)
{
	// memory mapping would be best
	memory_mapped_file mf;
	auto *buf = mf.map(fn);

	if (!buf)
		return mf;

	// i16 label (centipawns)
	// 116 outcome
	// i16 flags (bit 0 = turn)
	// u8 x 32 nibble-packed board
	// => 38 bytes per packed position

	size_t num_positions = mf.size() / PACKED_TRAIN_ENTRY_SIZE;

	printf("%I64u positions\n", num_positions);

	return mf;
#undef mem_read_int
#undef mem_read_buf
}

struct packed_network
{
	std::vector<float> weights;
	std::vector<float> biases;
};

struct network : torch::nn::Module
{
	torch::Tensor forward(torch::Tensor input, torch::Tensor input_opp);

	network();

	// pack: weights then biases
	packed_network pack() const;
	// unpack from packed network format
	void unpack(const packed_network &pn);

	void load_file(const char *fn);
	void save_file(const char *fn);
	void save_fixedpt_file(const char *fn);

	torch::nn::Linear layer0;
	torch::nn::Linear layer1;
	torch::nn::Linear layer2;

	std::vector<const torch::nn::Linear *> layers;
};

network::network()
	: layer0{INPUT_SIZE, cheng4::topo1}
	, layer1{cheng4::topo1in, cheng4::topoLayers >= 3 ? cheng4::topo2 : 1}
	, layer2{cheng4::topo2, 1}
{
	layers.push_back(&layer0);
	layers.push_back(&layer1);

	if (cheng4::topoLayers >= 3)
		layers.push_back(&layer2);

	register_module("layer0", layer0);
	register_module("layer1", layer1);

	if (cheng4::topoLayers >= 3)
		register_module("layer2", layer2);
}

void network::save_file(const char *fn)
{
	auto pn = pack();
	FILE *f = fopen(fn, "wb");
	fwrite(pn.weights.data(), sizeof(float), pn.weights.size(), f);
	fwrite(pn.biases.data(), sizeof(float), pn.biases.size(), f);
	fclose(f);
}

void network::save_fixedpt_file(const char *fn)
{
	auto pn = pack();

	std::vector<float> input;
	std::vector<int32_t> output;
	input.insert(input.end(), pn.weights.begin(), pn.weights.end());
	input.insert(input.end(), pn.biases.begin(), pn.biases.end());
	output.resize(input.size());

	// convert to 16:16 fixedpoint
	for (size_t i=0; i<output.size(); i++)
		output[i] = (int32_t)floor(input[i]*65536.0f + 0.5f);

	FILE *f = fopen(fn, "wb");
	fwrite(output.data(), sizeof(float), output.size(), f);
	fclose(f);
}

void network::load_file(const char *fn)
{
	auto pn = pack();
	FILE *f = fopen(fn, "rb");
	if (!f)
		return;

	fread(pn.weights.data(), sizeof(float), pn.weights.size(), f);
	fread(pn.biases.data(), sizeof(float), pn.biases.size(), f);
	fclose(f);

	auto all = pn.weights;
	all.insert(all.end(), pn.biases.begin(), pn.biases.end());

	float minw = +INFINITY;
	float maxw = -INFINITY;

	for (auto w : all)
	{
		minw = std::min<float>(w, minw);
		maxw = std::max<float>(w, maxw);
	}

	unpack(pn);

	printf("loaded network: min weight = %0.6lf max weight = %0.6lf\n", minw, maxw);
}

packed_network network::pack() const
{
	packed_network res;

	for (auto *it : layers)
	{
		auto &w = (*it)->weight;
		auto &b = (*it)->bias;
		auto wvec = unpack_tensor(w);
		auto bvec = unpack_tensor(b);
		res.weights.insert(res.weights.end(), wvec.begin(), wvec.end());
		res.biases.insert(res.biases.end(), bvec.begin(), bvec.end());
	}

	return res;
}

void network::unpack(const packed_network &pn)
{
	size_t wofs = 0;
	size_t bofs = 0;

	for (auto *it : layers)
	{
		auto &w = (*it)->weight;
		auto &b = (*it)->bias;

		auto wsize = tensor_size(w);
		auto bsize = tensor_size(b);

		pack_tensor(w, &pn.weights[wofs]);
		pack_tensor(b, &pn.biases[bofs]);

		wofs += wsize;
		bofs += bsize;
	}
}

torch::Tensor network::forward(torch::Tensor input, torch::Tensor input_opp)
{
	torch::Tensor tmp_std = relu(layer0->forward(input));
	torch::Tensor tmp_opp = relu(layer0->forward(input_opp));

	torch::Tensor tmp = torch::hstack({tmp_std, tmp_opp});

	if (cheng4::topoLayers >= 3)
	{
		tmp = relu(layer1->forward(tmp));
		tmp = layer2->forward(tmp);
	}
	else
	{
		tmp = layer1->forward(tmp);
	}

	return tmp;
}

// network trainer

struct net_trainer
{
	void train(memory_mapped_file &mf, size_t num_positions, network &net, int epochs = 50);

private:
	network *netref = nullptr;
};

void net_trainer::train(memory_mapped_file &mf, uint64_t num_positions, network &net, int epochs)
{
	netref = &net;

	auto device = at::kCUDA;
	constexpr auto cpudevice = at::kCPU;

	if (!torch::cuda::is_available())
	{
		printf("CUDA not available!\n");
		device = cpudevice;
	}

	//torch::optim::SGD optimizer(net.parameters(), 0.1);

	torch::optim::AdamOptions opts;
	auto lr = opts.get_lr();
	printf("adam default lr: %lf\n", lr);

	// Adam seems much better at converging
	torch::optim::Adam optimizer(net.parameters());

	const size_t num_batches = (size_t)(num_positions + BATCH_SIZE-1) / BATCH_SIZE;

	for (int epoch=0; epoch<epochs; epoch++)
	{
		printf("starting epoch %d, lr=%0.6lf\n", 1+epoch, lr);
		size_t idx = 0;

		auto cstart = clock();

		double loss_sum = 0.0;
		size_t batch_count = 0;

		net.to(device);

		// for each batch:
		for (size_t i=0; i<num_positions; i += BATCH_SIZE)
		{
			size_t count = std::min<size_t>(BATCH_SIZE, num_positions - i);

			// okay, now we must create batch tensor and fill it with data
			torch::Tensor input_batch = torch::zeros({(int)count, INPUT_SIZE});
			torch::Tensor input_batch_opp = torch::zeros({(int)count, INPUT_SIZE});

			torch::Tensor target = torch::zeros({(int)count, 1});

			float *itensor = static_cast<float *>(input_batch.mutable_data_ptr());
			float *itensor_opp = static_cast<float *>(input_batch_opp.mutable_data_ptr());
			float *ttensor = static_cast<float *>(target.mutable_data_ptr());

			#pragma omp parallel for
			for (int j=0; j<(int)count; j++)
			{
				auto *beg = mf.data() + (i+j)*PACKED_TRAIN_ENTRY_SIZE;
				auto *end = mf.data() + mf.size();
				auto lp = mem_load_position(beg, end);
				unpack_position_fast(&itensor[j*INPUT_SIZE], &itensor_opp[j*INPUT_SIZE], lp);
				ttensor[j] = label_position(lp);
			}

			input_batch = input_batch.to(device);
			input_batch_opp = input_batch_opp.to(device);
			target = target.to(device);

			optimizer.zero_grad();

			auto prediction = net.forward(input_batch, input_batch_opp);

			torch::Tensor loss = torch::mse_loss(::sigmoid(prediction*100.0f), ::sigmoid(target*100.0f));

			loss.backward();
			optimizer.step();

			auto batch_loss = loss.item<float>();

			if (idx++ % 5 == 0)
			{
				// print stuff
				std::cout << "Epoch: " << (epoch+1) << " | Batch: " << batch_count << "/" << num_batches
					<< " | Loss: " << batch_loss << " | Error: " << std::sqrt(batch_loss)*100 << "%" << std::endl;

				auto tc = clock();
				auto delta = tc - cstart;

				if (profile)
					printf("took %g sec\n", (double)delta / CLOCKS_PER_SEC);

				cstart = tc;

				net.to(cpudevice);
				net.save_file(NET_FILENAME);
				net.save_fixedpt_file(NET_FP_FILENAME);
				net.to(device);
			}

			loss_sum += batch_loss;
			++batch_count;
		}

		net.to(cpudevice);
		net.save_file(NET_FILENAME);
		net.save_fixedpt_file(NET_FP_FILENAME);

		printf("done_epoch %d: Loss %0.6lf | Error: %0.2lf%%\n", epoch+1, loss_sum / batch_count, std::sqrt(loss_sum / batch_count)*100);

		// lr epoch decay
		lr *= EPOCH_LR_DECAY_RATE;

		// reference: https://stackoverflow.com/questions/62415285/updating-learning-rate-with-libtorch-1-5-and-optimiser-options-in-c
		for (auto param_group : optimizer.param_groups())
			static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(lr);
	}

	netref = nullptr;
}

int main()
{
	// note: must be preshuffled
	auto mf = load_trainfile("autoplay.bin");

	network net;

	net.load_file(NET_FILENAME);

	net_trainer nt;
	// 50 epochs
	nt.train(mf, mf.size() / PACKED_TRAIN_ENTRY_SIZE, net, 50);

	return 0;
}
