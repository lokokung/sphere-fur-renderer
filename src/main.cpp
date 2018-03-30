#include "main.hpp"

#include "cereal/archives/json.hpp"
// #include "cereal/types/eigen.hpp"
#include "logger.hpp"

#include "../cuda/structs.hpp"

#include <fstream>

int main(int argc, char* argv[]) {
    // Initialize logging
    initialize_logging();
    INFO("Initialized logging library...");
    
    INFO("Terminating program...");
    return 0;
}
