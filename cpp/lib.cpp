

#include <stdio>
#include <iostream>
#include "lib.h"
int print(char* x) {
    std::string s(x);
    std::cout << s << std::endl;
    return s.size();
}
