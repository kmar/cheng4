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

#if defined(__APPLE__)
#	include <TargetConditionals.h>
#endif

#if (defined(__GNUC__) && (defined(__LP64__) || defined(__x86_64__) || defined(__aarch64__))) ||\
	(defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_X64)))
#	define CHENG_64BIT              1
#else
// note: just guessing here....
#	define CHENG_32BIT              1
#endif

#if (defined(__GNUC__) && defined(__x86_64__)) || (defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_X64)))
#	define CHENG_CPU_AMD64          1
#	define CHENG_CPU_X86            1
#elif (defined(__GNUC__) && (defined(__aarch64__) || defined(__arm64__) || defined(__arm__) || defined(__thumb__))) || \
	(defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARMT)))
#	define CHENG_CPU_ARM            1
#	if defined(__GNUC__) && (defined(__aarch64__) || defined(__arm64__))
#		define CHENG_CPU_ARM64      1
#	endif
#elif (defined(__GNUC__) && (defined(__i386__) || defined(__i386))) || (defined(_MSC_VER) && defined(_M_IX86))	\
	|| defined(_X86_) || defined(__X86__) || defined(__I86__)
#	define CHENG_CPU_X86            1
#endif

#if __AVX2__
#	define CHENG_CPU_AVX2           1
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#	define CHENG_CPU_ARM_NEON       1
#endif

#if defined(__clang__)
#	define CHENG_COMPILER_CLANG     1
#endif

#define CHENG_PTR_NOALIAS __restrict

#if CHENG_COMPILER_CLANG
#	define CHENG_AUTO_VECTORIZE_LOOP _Pragma("clang loop vectorize(enable)")
#else
#	define CHENG_AUTO_VECTORIZE_LOOP
#endif
