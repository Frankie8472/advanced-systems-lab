#include "common.h"

std::vector<compute_bw_func> *FuncRegister::user_funcs = nullptr;
std::vector<std::string> *FuncRegister::func_names = nullptr;
std::vector<std::string> *FuncRegister::func_descs = nullptr;
std::string FuncRegister::baseline_name = "";
compute_bw_func FuncRegister::baseline_func = nullptr;

void FuncRegister::add_function(compute_bw_func f, std::string name, std::string description){
    if(!user_funcs)
        user_funcs = new std::vector<compute_bw_func>();

    if(!func_names)
        func_names = new std::vector<std::string>();
    
    if(!func_descs)
        func_descs = new std::vector<std::string>();

    (*user_funcs).push_back(f);
    (*func_names).push_back(name);
    (*func_descs).push_back(description);
}

void FuncRegister::set_baseline(compute_bw_func f, std::string name){
    baseline_func = f;
    baseline_name = name;
}

void FuncRegister::printRegisteredFuncs()
{
    printf("Registered baseline is '%s'\n", baseline_name.c_str());
    printf("User functions:\n");
    for(size_t i = 0; i < size(); i++){
        printf("%20s: %s\n", (*func_names).at(i).c_str(), (*func_descs).at(i).c_str());
    }
}