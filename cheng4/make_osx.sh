clang++ -Wall -Wpedantic -W -O4 -std=c++0x -fno-stack-protector -fno-rtti -fno-exceptions -fomit-frame-pointer -DNDEBUG -U_FORTIFY_SOURCE -static allinone.cpp -o cheng4_osx_x64
strip cheng4_osx_x64
