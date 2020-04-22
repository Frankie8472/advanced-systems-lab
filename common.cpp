#include "common.h"

std::vector<compute_bw_func> *FuncRegister::user_funcs = nullptr;
std::vector<std::string> *FuncRegister::func_names = nullptr;

void FuncRegister::add_function(compute_bw_func f, std::string name){
    if(!user_funcs)
        user_funcs = new std::vector<compute_bw_func>();

    if(!func_names)
        func_names = new std::vector<std::string>();

    printf("Adding function '%s'\n", name.c_str());
    (*user_funcs).push_back(f);
    (*func_names).push_back(name);
}