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

#include "engine.h"
#include "protocol.h"
#include "utils.h"
#include <cstring>

int main( int argc, char **argv )
{
	// disable I/O buffering
	cheng4::disableIOBuffering();

	// static init
	cheng4::Engine::init( argc-1, const_cast<const char **>(argv)+1 );

	cheng4::Engine *eng = new cheng4::Engine;
	cheng4::Protocol *proto = new cheng4::Protocol( *eng );

	eng->run();

	// --cmd n is passed as protocol command
	for (int i=1; i<argc; i++)
		if (strcmp(argv[i], "--cmd") == 0 && i+1 < argc)
			proto->parse(argv[++i]);

	while ( !proto->shouldQuit() )
	{
		std::string line;
		cheng4::getline(line);
		proto->parse( line );
	}

	delete proto;
	delete eng;

	cheng4::Engine::done();
	return 0;
}
