#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <torch/torch.h>

// I know... dumb and lazy
#ifdef _DEBUG
#	pragma comment(lib, "c:/libtorch_d/lib/torch.lib")
#	pragma comment(lib, "c:/libtorch_d/lib/torch_cuda.lib")
#	pragma comment(lib, "c:/libtorch_d/lib/torch_cpu.lib")
#	pragma comment(lib, "c:/libtorch_d/lib/dnnl.lib")
#	pragma comment(lib, "c:/libtorch_d/lib/c10.lib")
#	pragma comment(lib, "c:/libtorch_d/lib/pytorch_jni.lib")
#else
#	pragma comment(lib, "c:/libtorch/lib/torch.lib")
#	pragma comment(lib, "c:/libtorch/lib/torch_cuda.lib")
#	pragma comment(lib, "c:/libtorch/lib/torch_cpu.lib")
#	pragma comment(lib, "c:/libtorch/lib/dnnl.lib")
#	pragma comment(lib, "c:/libtorch/lib/c10.lib")
#	pragma comment(lib, "c:/libtorch/lib/pytorch_jni.lib")
#endif
