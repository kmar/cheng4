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
#include "rnd_shuf.h"

#include "../cheng4/net.h"

// as big as we can fit into memory
constexpr int BATCH_SIZE = 1024*1024*1;
constexpr int INPUT_SIZE = cheng4::topo0;

// 1% per epoch
constexpr double EPOCH_LR_DECAY_RATE = 0.99;

constexpr char NET_FILENAME[] = "test.net";

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
	float eval;
	int16_t score;
	int16_t outcome;
	int16_t flags;
	int16_t count;
	std::vector<int16_t> indices;

	labeled_position() = default;

	labeled_position(const labeled_position &) = default;
	labeled_position(labeled_position &&) = default;

	labeled_position &operator =(const labeled_position &) = default;
	labeled_position &operator =(labeled_position &&) = default;
};

std::vector<labeled_position> positions;
// 10% validation
std::vector<labeled_position> validation_set;
// 10% test
std::vector<labeled_position> test_set;

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

void shuffle_positions(bool noseed = false)
{
	std::mt19937_64 rnd_engine;
	rnd_engine.seed(noseed ? (uint64_t)0 : seed_rnd());

	if (positions.empty())
		return;

	// fisher-yates shuffle
	for (size_t i=positions.size()-1; i>1; i--)
	{
		auto range = i-1;
		auto swidx = rnd_engine() % positions.size();
		std::swap(positions[i], positions[swidx]);
	}
}

void unpack_position_fast(void *dstp, const labeled_position &pos)
{
	auto *dst = static_cast<float *>(dstp);

	for (auto idx : pos.indices)
		dst[idx] = 1.0f;
}

void unpack_position(void *dstp, const labeled_position &pos)
{
	auto *dst = static_cast<float *>(dstp);

	for (int i=0; i<INPUT_SIZE; i++)
		dst[i] = 0.0f;

	unpack_position_fast(dst, pos);
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

void load_trainfile(const char *fn)
{
	positions.clear();
	validation_set.clear();
	test_set.clear();

	// memory mapping would be best

	memory_mapped_file mf;
	auto *buf = mf.map(fn);

	if (!buf)
		return;

	// i16 label (centipawns)
	// i16 static eval (centipawns)
	// 116 outcome
	// i16 flags
	// i16 count
	// count * i16 index

	const uint8_t *ptr = buf;
	const uint8_t *end = buf + mf.size();

#define mem_read_int(var) \
	if (ptr + sizeof(var) > end) \
		break; \
	memcpy(&(var), ptr, sizeof(var)); \
	ptr += sizeof(var)

	for (;;)
	{
		int16_t tmp;

		mem_read_int(tmp);

		labeled_position p;
		p.score = tmp;
		p.label = tmp/100.0f;

		mem_read_int(tmp);

		p.eval = tmp/100.0f;

		mem_read_int(p.outcome);
		mem_read_int(p.flags);

		bool blackToMove = (p.flags & 1) != 0;

		// also: indices are from stm's point of view!
		if (blackToMove)
			p.outcome = 2 - p.outcome;

		mem_read_int(p.count);

		p.indices.resize(p.count);

		if (p.count)
		{
			auto bytes = p.count*sizeof(int16_t);

			if (ptr + bytes > end)
				break;

			memcpy(p.indices.data(), ptr, bytes);
			ptr += bytes;
		}

		positions.emplace_back(std::move(p));

#ifdef _DEBUG
		if (positions.size() >= 2'000'000)
			break;
#endif
	}

	// deterministic shuffle
	shuffle_positions(/*noseed*/true);

	printf("%I64d positions loaded\n", (int64_t)positions.size());

	// 10%
	size_t test_size = positions.size()/10;

	validation_set.insert(validation_set.end(), positions.end() - test_size, positions.end());
	positions.resize(positions.size() - test_size);
	test_set.insert(test_set.end(), positions.end() - test_size, positions.end());
	positions.resize(positions.size() - test_size);

	// align positions to batch size
	positions.resize(positions.size() - positions.size() % BATCH_SIZE);

	printf("%I64d positions reserved for validation\n", (int64_t)validation_set.size());
	printf("%I64d positions reserved for testing\n", (int64_t)test_set.size());
	printf("%I64d positions reserved for training\n", (int64_t)positions.size());
#undef mem_read_int
}

struct packed_network
{
	std::vector<float> weights;
	std::vector<float> biases;
};

struct network : torch::nn::Module
{
	torch::Tensor forward(torch::Tensor input);

	network();

	// pack: weights then biases
	packed_network pack() const;
	// unpack from packed network format
	void unpack(const packed_network &pn);

	void load_file(const char *fn);
	void save_file(const char *fn);

	torch::nn::Linear layer0;
	torch::nn::Linear layer1;
	torch::nn::Linear layer2;

	const torch::nn::Linear *layers[3] =
	{
		&layer0,
		&layer1,
		&layer2
	};
};

network::network()
	: layer0{INPUT_SIZE, cheng4::topo1}
	, layer1{cheng4::topo1, cheng4::topo2}
	, layer2{cheng4::topo2, 1}
{
	register_module("layer0", layer0);
	register_module("layer1", layer1);
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

torch::Tensor fixed_leaky_relu(torch::Tensor input)
{
	torch::nn::functional::LeakyReLUFuncOptions opts;
	opts.negative_slope(0.01);
	torch::Tensor res = torch::nn::functional::leaky_relu(input, opts);
	return res;
}

torch::Tensor network::forward(torch::Tensor input)
{
	torch::Tensor tmp = fixed_leaky_relu(layer0->forward(input));
	tmp = fixed_leaky_relu(layer1->forward(tmp));
	tmp = layer2->forward(tmp);
	return tmp;
}

// network trainer

struct net_trainer
{
	void train(network &net, int epochs = 50);

private:
	network *netref = nullptr;
};

void net_trainer::train(network &net, int epochs)
{
	netref = &net;

	// somehow, CUDA crashes libtorch...
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

	const size_t num_batches = (positions.size() + BATCH_SIZE-1) / BATCH_SIZE;

	for (int epoch=0; epoch<epochs; epoch++)
	{
		shuffle_positions();
		printf("starting epoch %d, lr=%0.6lf\n", 1+epoch, lr);
		size_t idx = 0;

		auto cstart = clock();

		double loss_sum = 0.0;
		size_t batch_count = 0;

		net.to(device);

		// for each batch:
		for (size_t i=0; i<positions.size(); i += BATCH_SIZE)
		{
			size_t count = std::min<size_t>(BATCH_SIZE, positions.size() - i);

			// okay, now we must create batch tensor and fill it with data
			torch::Tensor input_batch = torch::zeros({(int)count, INPUT_SIZE});

			torch::Tensor target = torch::zeros({(int)count, 1});

			float *itensor = static_cast<float *>(input_batch.mutable_data_ptr());
			float *ttensor = static_cast<float *>(target.mutable_data_ptr());

			#pragma omp parallel for
			for (int j=0; j<(int)count; j++)
			{
				unpack_position_fast(&itensor[j*INPUT_SIZE], positions[i+j]);
				ttensor[j] = label_position(positions[i+j]);
			}

			input_batch = input_batch.to(device);
			target = target.to(device);

			optimizer.zero_grad();

			auto prediction = net.forward(input_batch);

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
				net.to(device);
			}

			loss_sum += batch_loss;
			++batch_count;
		}

		net.to(cpudevice);
		net.save_file(NET_FILENAME);

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
	load_trainfile("labelFEN_out.bin");

	network net;

	net.load_file(NET_FILENAME);

	net_trainer nt;
	// 50 epochs
	nt.train(net, 50);

	size_t test_set_size = test_set.size();

	cheng4::Network cnet;
	const int sizes[] = {cheng4::topo0, cheng4::topo1, cheng4::topo2, 1};

	cnet.init_topology(sizes, 4);
	cnet.load(NET_FILENAME);
	cnet.transpose_weights();

	for (size_t i=0; i<std::min<size_t>(1000, test_set.size()); i++)
	{
		const auto &p = test_set[i];
		torch::Tensor test = torch::zeros(INPUT_SIZE);
		unpack_position(test.mutable_data_ptr(), p);

		float outp[1];
		cheng4::i32 nz[INPUT_SIZE];

		for (size_t i=0; i<p.indices.size(); i++)
			nz[i] = p.indices[i];

		cnet.forward_nz(static_cast<const float *>(test.data_ptr()), INPUT_SIZE,
			nz, (int)p.indices.size(), outp, 1);

		auto inf = net.forward(test);
		auto utensor = unpack_tensor(inf);

		printf("test position %d\n", (int)i);

		printf("\tinferred_value: %0.4lf\n", utensor[0]);
		printf("\tCHENG: inferred_value: %0.4lf\n", outp[0]);
		printf("\tlabel: %0.4lf\n", label_position(p));
		printf("\terror: %0.4lf\n", std::abs(label_position(p) - utensor[0]));
	}

	return 0;
}
