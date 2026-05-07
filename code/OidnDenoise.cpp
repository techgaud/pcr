#include "Includes/OidnDenoise.h"

#ifdef PCR_HAVE_OIDN
#include <iostream>
#include <mutex>
#include <OpenImageDenoise/oidn.hpp>
#endif

namespace OidnDenoise
{
    bool isAvailable()
    {
#ifdef PCR_HAVE_OIDN
        return true;
#else
        return false;
#endif
    }

    bool denoise(std::vector<Vec3f> &color,
                 const std::vector<Vec3f> &albedo,
                 const std::vector<Vec3f> &normal,
                 int width, int height)
    {
#ifdef PCR_HAVE_OIDN
        if (width <= 0 || height <= 0)
            return false;
        if ((int)color.size() != width * height)
            return false;

        // Lazy single-instance device. OIDN device init is non-trivial
        // (probes for accelerators, JIT compiles kernels); we want to do
        // it once per process, not once per render.
        static std::mutex deviceMu;
        static oidn::DeviceRef device;
        static bool deviceReady = false;
        {
            std::lock_guard<std::mutex> lk(deviceMu);
            if (!deviceReady)
            {
                device = oidn::newDevice();
                device.commit();
                const char *err = nullptr;
                if (device.getError(err) != oidn::Error::None)
                {
                    std::cerr << "OIDN device init failed: "
                              << (err ? err : "(no message)") << std::endl;
                    return false;
                }
                deviceReady = true;
            }
        }

        oidn::FilterRef filter = device.newFilter("RT");
        filter.setImage("color", color.data(), oidn::Format::Float3,
                        (size_t)width, (size_t)height);
        if ((int)albedo.size() == width * height)
        {
            filter.setImage("albedo",
                            const_cast<Vec3f *>(albedo.data()),
                            oidn::Format::Float3,
                            (size_t)width, (size_t)height);
        }
        if ((int)normal.size() == width * height)
        {
            filter.setImage("normal",
                            const_cast<Vec3f *>(normal.data()),
                            oidn::Format::Float3,
                            (size_t)width, (size_t)height);
        }
        filter.setImage("output", color.data(), oidn::Format::Float3,
                        (size_t)width, (size_t)height);
        filter.set("hdr", true);
        filter.commit();
        filter.execute();

        const char *err = nullptr;
        if (device.getError(err) != oidn::Error::None)
        {
            std::cerr << "OIDN filter error: "
                      << (err ? err : "(no message)") << std::endl;
            return false;
        }
        return true;
#else
        (void)color; (void)albedo; (void)normal; (void)width; (void)height;
        return false;
#endif
    }
}
