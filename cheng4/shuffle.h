#pragma once

#include "chtypes.h"

namespace cheng4
{

using ULong = u64;
using Byte = u8;
using UInt = u32;
using UIntPtr = uintptr_t;
using Int = i32;
using UShort = u16;

// my simple KISS PRNG
class FastRandom
{
	inline ULong Rotate(ULong v, Byte s)
	{
		return (v >> s) | (v << (64-s));
	}
public:
	explicit FastRandom(ULong initialSeed = 0)
	{
		Seed(initialSeed);
	}

	// generate next 64-bit random number
	inline ULong Next64()
	{
		ULong tmp = keys[0];
		keys[0] += Rotate(keys[1] ^ 0xc5462216u ^ ((ULong)0xcf14f4ebu<<32), 1);
		return keys[1] += Rotate(tmp ^ 0x75ecfc58u ^ ((ULong)0x9576080cu<<32), 9);
	}

	inline UInt Next()
	{
		return (UInt)Next64();
	}

	// seed (this time 64-bit)
	void Seed(ULong val)
	{
		keys[0] = keys[1] = val;

		for (int i=0; i<64; i++)
			Next();
	}

	// note: made public so we can serialize
	ULong keys[2];
};

// note: 2**result may be less that v!
// actually, this is MSBit index
template< typename T > static inline UShort Log2Int(T v)
{
	assert(v > 0);
	// FIXME: better! can use bit scan on x86/x64
	UShort res = 0;

	while (v > 0)
	{
		res++;
		v >>= 1;
	}

	return res-1;
}

// this returns x such that (1 << x) >= v
// don't pass
template< typename T > static inline UShort Log2Size(T v)
{
	UShort res = Log2Int(v);
	return res += ((T)1 << res) < v;
}

// uniform random distribution for (unsigned!!) integers
template< typename T >
class UniformDist
{
public:
	UniformDist(ULong from, ULong to, T &gen) : generator(gen), accum(0)
	{
		Clear();
		SetRange(from, to);
	}

	inline void Clear()
	{
		accumBits = 0;
	}

	void SetRange(ULong from, ULong to)
	{
		if (from > to)
			std::swap(from, to);

		minv = from;
		delta = to - from + 1;
		assert(delta != 0);
		nbits = Log2Size(delta);
		mask = ((ULong)1 << nbits)-1;
	}

	ULong Next64()
	{
		ULong res;

		for(;;)
		{
			if (accumBits < nbits)
			{
				accum = generator.Next64();
				accumBits = sizeof(ULong)*8;
			}

			res = accum & mask;
			accumBits -= nbits;
			accum >>= nbits;

			if (res < delta)
				break;
		}

		return minv + res;
	}
	inline UInt Next()
	{
		return (UInt)Next64();
	}
private:
	T &generator;
	ULong minv;
	ULong delta;
	ULong accum;
	ULong mask;
	UShort nbits;
	Int accumBits;
};

template<typename T, typename R>
void ShuffleArray(T *ptr, const T *top, R &rng)
{
	UniformDist<R> udist(0, 0, rng);
	// performs Fisher-Yates shuffle
	assert(top >= ptr);
	UIntPtr d = (UIntPtr)(top - ptr);

	for (UIntPtr i=d; i > 1; i--)
	{
		UIntPtr cur = i-1;
		UIntPtr j;
		udist.SetRange(0, cur);

		if (sizeof(UIntPtr) <= sizeof(UInt))
			j = (UIntPtr)udist.Next();
		else
			j = (UIntPtr)udist.Next64();

		std::swap(ptr[cur], ptr[j]);
	}
}

}
