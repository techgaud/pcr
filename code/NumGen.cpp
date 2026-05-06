#include "Includes/NumGen.h"

float NumGen::Epsilon()
{
    thread_local std::mt19937 gen{std::random_device{}()};
    thread_local std::uniform_real_distribution<float> distribution = std::uniform_real_distribution<float>(0.f, 1.f);
    return distribution(gen);
};
