#include <vector>
#include <iostream>

#include "Includes/Renderer.h"
#include "Includes/Sphere.h"

int main()
{
    int depth, samples, shadowSamples;
    std::cout << "Enter depth: " << std::endl;
    std::cin >> depth;
    std::cout << "Enter samples: " << std::endl;
    std::cin >> samples;
    std::cout << "Enter shadow samples: " << std::endl;
    std::cin >> shadowSamples;

    
    Material nonemissive{Vec3f(0.4, 0.4, 0.3), Vec3f(0.f, 0.f, 0.f)};
    Material emissive{Vec3f(0.f, 0.f, 0.f), Vec3f{1.0f, 0.85f, 0.6f}};
    emissive.emissive *= 80.f;   // brightness
    
    Plane lightSource(Vec3f{-0.375f, 2.f, -4.25f}, Vec3f{0.75f, 0, 0}, Vec3f{0, 0, 0.4f}, emissive);    // small
    // Plane lightSource(Vec3f{-1., 2.f, -4.25f}, Vec3f{2.f, 0, 0}, Vec3f{0, 0, 0.4f}, emissive);          // large
    
    std::vector<Sphere> spheres{
        Sphere(Vec3f(0.f, -1.f, -4.5f), 0.75f, nonemissive)
        // Sphere(Vec3f(0.f, 2.25f, -4.f), 0.5f, nonemissive)
    };
    
    Renderer renderer{712, 712, 65.f, depth, samples, shadowSamples, lightSource};
    renderer.render(spheres);

    return 0;
}