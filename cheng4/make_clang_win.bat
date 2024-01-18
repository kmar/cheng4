clang++ -m64 -Wall -Wpedantic -W -O3 -std=c++14 -fno-stack-protector -fomit-frame-pointer -fno-rtti -fno-exceptions -DNDEBUG -U_FORTIFY_SOURCE -static allinone.cpp -o cheng4_x64.exe -lwinmm
clang++ -march=core-avx2 -m64 -Wall -Wpedantic -W -O3 -std=c++14 -fno-stack-protector -fomit-frame-pointer -fno-rtti -fno-exceptions -DNDEBUG -U_FORTIFY_SOURCE -static allinone.cpp -o cheng4_avx2.exe -lwinmm
