/*
   mini-LZ library (mlz)
   (c) Martin Sedlak 2016-2018

   Boost Software License - Version 1.0 - August 17th, 2003

   Permission is hereby granted, free of charge, to any person or organization
   obtaining a copy of the software and accompanying documentation covered by
   this license (the "Software") to use, reproduce, display, distribute,
   execute, and transmit the Software, and to prepare derivative works of the
   Software, and to permit third-parties to whom the Software is furnished to
   do so, all subject to the following:

   The copyright notices in the Software and this entire statement, including
   the above license grant, this restriction and the following disclaimer,
   must be included in all copies of the Software, in whole or in part, and
   all derivative works of the Software, unless such copies or derivative
   works are solely in the form of machine-executable object code generated by
   a source language processor.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
   SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
   FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#ifndef MLZ_DEC_MINI_H
#define MLZ_DEC_MINI_H

/* unsafe (=no bounds checks) minimal all-in-one decompression */
/* define MLZ_DEC_MINI_IMPLEMENTATION to include implementation */

#if !defined(MLZ_API)
#	define MLZ_API
#endif

MLZ_API int
mlz_decompress_mini(
	void       *dst,
	const void *src,
	int         src_size
);

#if defined(MLZ_DEC_MINI_IMPLEMENTATION)

#include <string.h>

#if !defined(MLZ_COMMON_H)

#if (defined(_MSC_VER) && _MSC_VER < 1900) || defined(__BORLANDC__)
typedef signed   __int8  mlz_sbyte;
typedef unsigned __int8  mlz_byte;
typedef unsigned __int32 mlz_uint;
typedef signed   __int32 mlz_int;
#else
#	include <stdint.h>
typedef int8_t      mlz_sbyte;
typedef uint8_t     mlz_byte;
typedef uint32_t    mlz_uint;
typedef int32_t     mlz_int;
#endif

#if !defined(MLZ_CONST)
#	define MLZ_CONST const
#endif

#define MLZ_MIN_MATCH   3
#define MLZ_ACCUM_BITS  24
#define MLZ_ACCUM_BYTES ((MLZ_ACCUM_BITS)/8)
#define MLZ_MIN_LIT_RUN 23

/* !defined MLZ_COMMON_H */
#endif

#define MLZ_DEC_GUARD_MASK (1u << MLZ_ACCUM_BITS)
#define MLZ_DEC_0BIT_MASK  ~1u
#define MLZ_DEC_2BIT_MASK  ~7u
#define MLZ_DEC_3BIT_MASK  ~15u
#define MLZ_DEC_6BIT_MASK  ~127u

#define MLZ_LOAD_ACCUM() \
	{ \
		int i; \
		accum  = MLZ_DEC_GUARD_MASK | *sb++; \
		for (i=1; i<MLZ_ACCUM_BYTES; i++) \
			accum += (mlz_uint)(*sb++) << (8*i); \
	}

#define MLZ_GET_BIT_FAST_NOACCUM(res) \
	res = (int)(accum & 1); \
	accum >>= 1;

#define MLZ_GET_BIT_FAST(res) \
	MLZ_GET_BIT_FAST_NOACCUM(res) \
	if (accum <= 1) { \
		MLZ_LOAD_ACCUM() \
	}

#define MLZ_GET_TYPE_FAST_NOACCUM(res) \
	res = (int)(accum & 3); \
	accum >>= 2;

#define MLZ_GET_TYPE_FAST(res) \
	if (accum & MLZ_DEC_2BIT_MASK) { \
		MLZ_GET_TYPE_FAST_NOACCUM(res) \
	} else { \
		int tmp; \
		MLZ_GET_BIT_FAST(res); \
		MLZ_GET_BIT_FAST(tmp); \
		res += 2*tmp; \
	}

#define MLZ_GET_SHORT_LEN_FAST_NOACCUM(res) \
	res = (int)(accum & 7); \
	accum >>= 3;

#define MLZ_GET_SHORT_LEN_FAST(res) \
	if (accum & MLZ_DEC_3BIT_MASK) { \
		MLZ_GET_SHORT_LEN_FAST_NOACCUM(res) \
	} else { \
		int tmp; \
		MLZ_GET_BIT_FAST(res) \
		MLZ_GET_BIT_FAST(tmp) \
		res += 2*tmp; \
		MLZ_GET_BIT_FAST(tmp) \
		res += 4*tmp; \
	}

#define MLZ_COPY_MATCH_UNSAFE() \
	chlen = (len+7) >> 3; \
	len &= 7; \
	dist = -dist; \
	if (dist > -8) { \
		int i; \
		for (i=0; i<8; i++) \
			db[i] = db[i+dist]; \
		db += 8; \
		--chlen; \
		dist = mlz_offset_table[-dist]; \
	} \
	while (chlen-- > 0) { \
		memcpy(db, db+dist, 8); \
		db += 8; \
	} \
	db -= (8-len) & 7;

#define MLZ_LITCOPY(db, sb, run) \
	{ \
		mlz_int chrun = run >> 2; \
		run &= 3; \
		while (chrun-- > 0) { \
			*db++ = *sb++; \
			*db++ = *sb++; \
			*db++ = *sb++; \
			*db++ = *sb++; \
		} \
 \
		while (run-- > 0) \
			*db++ = *sb++; \
	}

#define MLZ_LITERAL_RUN_UNSAFE() \
	{ \
		mlz_int run = *sb++; \
		if (len > MLZ_MIN_MATCH) \
			run += *sb++ << 8; \
		run += MLZ_MIN_LIT_RUN; \
		MLZ_LITCOPY(db, sb, run); \
	}

#define MLZ_TINY_MATCH() \
	len += MLZ_MIN_MATCH; \
	dist = *sb++;

#define MLZ_SHORT_MATCH() \
	dist = sb[0] + (sb[1] << 8); \
	sb += 2; \
	len = dist >> 13; \
	len += MLZ_MIN_MATCH; \
	dist &= (1 << 13) - 1;

#define MLZ_SHORT2_MATCH() \
	len += MLZ_MIN_MATCH; \
	dist = sb[0] + (sb[1] << 8); \
	sb += 2;

#define MLZ_FULL_MATCH() \
	len = sb[0]; \
	if (len == 255) { \
		len = sb[1] + (sb[2] << 8); \
		sb += 2; \
	} \
	len += MLZ_MIN_MATCH; \
	dist = sb[1] + (sb[2] << 8); \
	sb += 3;

#define MLZ_LITERAL_UNSAFE() \
	if (!bit0) { \
		*db++ = *sb++; \
		continue; \
	}

#define MLZ_INIT_DECOMPRESS() \
	mlz_uint accum; \
	mlz_int chlen; \
	int bit0, type; \
 \
	MLZ_CONST mlz_sbyte mlz_offset_table[] = {0, -8, -8, -9, -8, -10, -12, -14}; \
	MLZ_CONST mlz_byte *sb = (MLZ_CONST mlz_byte *)(src); \
	MLZ_CONST mlz_byte *se = sb + src_size; \
	mlz_byte *db = (mlz_byte *)dst; \
	MLZ_CONST mlz_byte *odb = db;

MLZ_API int
mlz_decompress_mini(
	void       *dst,
	const void *src,
	int         src_size
)
{
	MLZ_INIT_DECOMPRESS()
	mlz_int dist = 0, len = 0;
	(void)dist;
	(void)len;

	MLZ_LOAD_ACCUM()

	while (sb < se) {
		if ((accum & MLZ_DEC_6BIT_MASK)) {
			MLZ_GET_BIT_FAST_NOACCUM(bit0)
			MLZ_LITERAL_UNSAFE()

			/* match... */
			MLZ_GET_TYPE_FAST_NOACCUM(type)
			if (type == 0) {
				/* tiny match */
				MLZ_GET_SHORT_LEN_FAST_NOACCUM(len)
				MLZ_TINY_MATCH()
				if (dist == 0) {
					/* literal run */
					MLZ_LITERAL_RUN_UNSAFE()
					continue;
				}
			} else if (type == 2) {
				/* short match */
				MLZ_SHORT_MATCH()
			} else if (type == 1) {
				/* short2 match */
				MLZ_GET_SHORT_LEN_FAST_NOACCUM(len)
				MLZ_SHORT2_MATCH()
			} else {
				/* full match */
				MLZ_FULL_MATCH()
			}
			/* copy match */
			MLZ_COPY_MATCH_UNSAFE()
			continue;
		}

		MLZ_GET_BIT_FAST(bit0)
		MLZ_LITERAL_UNSAFE()

		/* match... */
		MLZ_GET_TYPE_FAST(type)
		if (type == 0) {
			/* tiny match */
			MLZ_GET_SHORT_LEN_FAST(len)
			MLZ_TINY_MATCH()
			if (dist == 0) {
				/* literal run  */
				MLZ_LITERAL_RUN_UNSAFE()
				continue;
			}
		} else if (type == 2) {
			/* short match */
			MLZ_SHORT_MATCH()
		} else if (type == 1) {
			/* short2 match */
			MLZ_GET_SHORT_LEN_FAST(len)
			MLZ_SHORT2_MATCH()
		} else {
			/* full match */
			MLZ_FULL_MATCH()
		}
		/* copy match */
		MLZ_COPY_MATCH_UNSAFE()
	}
	return (int)(db - odb);
}

#undef MLZ_DEC_GUARD_MASK
#undef MLZ_DEC_0BIT_MASK
#undef MLZ_DEC_2BIT_MASK
#undef MLZ_DEC_3BIT_MASK
#undef MLZ_DEC_6BIT_MASK
#undef MLZ_LOAD_ACCUM
#undef MLZ_GET_BIT_FAST_NOACCUM
#undef MLZ_GET_BIT_FAST
#undef MLZ_GET_TYPE_FAST_NOACCUM
#undef MLZ_GET_TYPE_FAST
#undef MLZ_GET_SHORT_LEN_FAST_NOACCUM
#undef MLZ_GET_SHORT_LEN_FAST
#undef MLZ_COPY_MATCH_UNSAFE
#undef MLZ_LITCOPY
#undef MLZ_LITERAL_RUN_UNSAFE
#undef MLZ_TINY_MATCH
#undef MLZ_SHORT_MATCH
#undef MLZ_SHORT2_MATCH
#undef MLZ_FULL_MATCH
#undef MLZ_LITERAL_UNSAFE
#undef MLZ_INIT_DECOMPRESS

#if !defined(MLZ_COMMON_H)
#	undef MLZ_MIN_MATCH
#	undef MLZ_ACCUM_BITS
#	undef MLZ_ACCUM_BYTES
#	undef MLZ_MIN_LIT_RUN
#	undef MLZ_CONST
#endif

#endif    /* !MLZ_DEC_MINI_IMPLEMENTATION */

#endif