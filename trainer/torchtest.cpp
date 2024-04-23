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

#include "../cheng4/shuffle.h"
#include "rnd_shuf.h"

constexpr int PACKED_TRAIN_ENTRY_SIZE = 28;

// set to true to random-shuffle batches before each epoch
constexpr bool SHUFFLE_BATCHES = true;

// as big as we can fit into memory
constexpr int BATCH_SIZE = 1024*1024/2;
constexpr int INPUT_SIZE = cheng4::topo0;

// 1% per epoch
constexpr double EPOCH_LR_DECAY_RATE = 0.99;

constexpr char NET_FILENAME[] = "test.net";
constexpr char NET_FP_FILENAME[] = "test.fpnet";
constexpr char NET_FP_FILENAME_EPOCH[] = "test_epoch%d.fpnet";

// last cheng HCE K for texel tuning
constexpr double HCE_K = 1.25098;

// profile batch time?
constexpr bool profile = false;

std::string epoch_filename(int epoch)
{
	char buffer[256];
	sprintf(buffer, NET_FP_FILENAME_EPOCH, epoch+1);

	return buffer;
}

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
	uint64_t occupancy;
	uint8_t pieces[16];
	int16_t score;
	int8_t outcome;
	int8_t flags;
	float label;

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
	auto count = (int16_t)netIndices(blackToMove, pos.occupancy, pos.pieces, ninds);

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

		// using nibble-packed indices
		mem_read_buf(&p.occupancy, 8);
		mem_read_buf(p.pieces, 16);
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
	// i8 outcome
	// i8 flags (bit 0 = turn)
	// u64 occupancy
	// u8 x 16 nibble-packed board
	// => 28 bytes per packed position

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

	void clamp_weights();

	static torch::Tensor activate(torch::Tensor t);
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
	std::vector<int16_t> output;
	input.insert(input.end(), pn.weights.begin(), pn.weights.end());
	input.insert(input.end(), pn.biases.begin(), pn.biases.end());
	output.resize(input.size());

	// convert to 7:9 fixedpoint
	for (size_t i=0; i<output.size(); i++)
		output[i] = (int16_t)floor(input[i]*512.0f + 0.5f);

	FILE *f = fopen(fn, "wb");
	fwrite(output.data(), sizeof(int16_t), output.size(), f);
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

void network::clamp_weights()
{
	torch::NoGradGuard guard;

	// 32767/65 = 504 (=bias + all squares occupied for netcache fit in 16-bit)
	constexpr float weight_limit = 504.0f/512.0f;

	for (auto *it : layers)
	{
		// (_ version = inplace)
		it->get()->weight.clamp_(-weight_limit, weight_limit);
		it->get()->bias.clamp_(-weight_limit, weight_limit);
	}
}

torch::Tensor network::activate(torch::Tensor t)
{
	// saturate aka clipped relu (_ version = inplace)
	return torch::clamp_(t, 0.0f, 1.0f);
}

torch::Tensor network::forward(torch::Tensor input, torch::Tensor input_opp)
{
	torch::Tensor tmp_std = activate(layer0->forward(input));
	torch::Tensor tmp_opp = activate(layer0->forward(input_opp));

	torch::Tensor tmp = torch::hstack({tmp_std, tmp_opp});

	if (cheng4::topoLayers >= 3)
	{
		tmp = activate(layer1->forward(tmp));
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

	std::vector<size_t> shuffled_batches;

	cheng4::FastRandom shuf_rng;

	if constexpr (SHUFFLE_BATCHES)
	{
		shuffled_batches.reserve(num_batches);

		for (size_t i=0; i<num_positions; i += BATCH_SIZE)
			shuffled_batches.push_back(i);

		shuf_rng.Seed(seed_rnd());
	}


	for (int epoch=0; epoch<epochs; epoch++)
	{
		printf("starting epoch %d, lr=%0.6lf\n", 1+epoch, lr);
		size_t idx = 0;


		if constexpr (SHUFFLE_BATCHES)
			cheng4::ShuffleArray(shuffled_batches.data(), shuffled_batches.data() + shuffled_batches.size(), shuf_rng);

		auto cstart = clock();

		double loss_sum = 0.0;
		size_t batch_count = 0;

		net.to(device);

		// for each batch:
		for (size_t bi=0; bi<num_positions; bi += BATCH_SIZE)
		{
			size_t i;

			if constexpr (SHUFFLE_BATCHES)
				i = shuffled_batches[idx];
			else
				i = bi;

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

			net.clamp_weights();

			auto batch_loss = loss.item<float>();

			if (idx++ % 5 == 0)
			{
				// print stuff
				printf("Epoch: %d | Batch: [src %-5d] %d/%d (%0.2lf%%) | Loss: %0.6lf | Error: %0.6lf\n",
					(int)(epoch+1),
					(int)(i / BATCH_SIZE),
					(int)batch_count,
					(int)num_batches,
					batch_count*100.0/num_batches,
					batch_loss,
					std::sqrt(batch_loss)*100
				);

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
		auto epochfn = epoch_filename(epoch);
		net.save_fixedpt_file(epochfn.c_str());

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
	// 50 epochs, overkill as data grows
	nt.train(mf, mf.size() / PACKED_TRAIN_ENTRY_SIZE, net, 50);

	return 0;
}
