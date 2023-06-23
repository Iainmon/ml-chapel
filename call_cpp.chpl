// use SysCTypes;
require "cpp/lib.h", "cpp/lib.c";

extern proc print(arg:c_string);


print("hello from chapel".c_str());