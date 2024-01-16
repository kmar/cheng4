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

#ifdef _MSC_VER
#	define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int pack_net(const char *filename, const char *outfilename)
{
	FILE *f = fopen(filename, "rb");

	if (!f)
	{
		fprintf(stderr, "couldn't open %s\n", filename);
		return 1;
	}

	fseek(f, 0, SEEK_END);
	long fsz = ftell(f);
	fseek(f, 0, SEEK_SET);

	long dwcount = (fsz+3)/4;
	long fsz4 = dwcount*4;

	uint8_t *inbuf = (uint8_t *)calloc((size_t)fsz4, 1);

	if (fread(inbuf, 1, fsz, f) != fsz)
	{
		free(inbuf);
		fclose(f);
		fprintf(stderr, "failed to read %s\n", filename);
		return 2;
	}

	fclose(f);

	FILE *f2 = fopen(outfilename, "w");

	if (!f2)
	{
		free(inbuf);
		fprintf(stderr, "failed to create %s\n", outfilename);
		return 3;
	}

	fprintf(f2, "const int NET_DATA_SIZE = %d;\n", (int)fsz);

	fprintf(f2, "const uint32_t NET_DATA[] = {\n");

	const uint8_t *src = inbuf;

	for (long i=0; i<dwcount; i++)
	{
		uint32_t tmp = 0;
		memcpy(&tmp, src, 4);
		src += 4;
		fprintf(f2, "0x%08x", tmp);

		if (i+1 < dwcount)
			fprintf(f2, ",");

		if ((i & 15) == 15)
			fprintf(f2, "\n");
	}

	fprintf(f2, "};\n");

	free(inbuf);
	fclose(f2);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf("usage: net_pak <infile>\n");
		return 0;
	}

	return pack_net(argv[1], "net_embed.h");
}
