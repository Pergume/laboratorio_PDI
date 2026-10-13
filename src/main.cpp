#include "frontend.hpp"

#include <exception>
#include <iostream>

int main() {
    try {
        return pdi::runFrontend();
    } catch (const std::exception& error) {
        std::cerr << "Erro fatal: " << error.what() << '\n';
        return 1;
    }
}
