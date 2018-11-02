#include <iostream>
#include <cstdio>
#pragma once
#if EVOLUTIONARY_DEBUG
#define DMESSAGE(x) (std::cout << (x))
#define DERROR(...) fprintf(stderr, __VA_ARGS__)
#else
#define DMESSAGE(x)
#define DERROR(...)
#endif
