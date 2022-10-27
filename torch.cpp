//
// Created by Nghĩa Khang on 19/10/2022.
//

#include <torch/torch.h>

#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
}