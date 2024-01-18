clang++ -Wall -Wpedantic -W -O3 -std=c++0x -fno-stack-protector -fno-rtti -fno-exceptions -fomit-frame-pointer -DNDEBUG -U_FORTIFY_SOURCE allinone.cpp -o cheng4_osx_arm64
strip cheng4_osx_arm64
