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

        // OIDN 2.x requires images to be backed by device-allocated
        // OIDNBuffers (not raw CPU std::vector pointers) when the
        // device backend can directly access GPU memory. The earlier
        // raw-pointer setImage() call silently failed every render on
        // Apple Silicon with:
        //   'image data not accessible by the device, please use
        //    OIDNBuffer or device allocator for storage'
        // and the filter.execute() never actually denoised anything.
        //
        // On Apple Silicon unified memory, the buf.write() / buf.read()
        // copies are essentially free since the OIDNBuffer is in the
        // same physical RAM as the source std::vector.
        const size_t bytes = (size_t)width * (size_t)height * 3 * sizeof(float);
        oidn::BufferRef colorBuf  = device.newBuffer(bytes);
        oidn::BufferRef albedoBuf;
        oidn::BufferRef normalBuf;
        if (!colorBuf)
        {
            std::cerr << "OIDN buffer alloc failed (color, "
                      << bytes << " bytes)" << std::endl;
            return false;
        }
        colorBuf.write(0, bytes, color.data());
        const bool haveAlbedo = (int)albedo.size() == width * height;
        const bool haveNormal = (int)normal.size() == width * height;
        if (haveAlbedo)
        {
            albedoBuf = device.newBuffer(bytes);
            if (albedoBuf) albedoBuf.write(0, bytes, albedo.data());
        }
        if (haveNormal)
        {
            normalBuf = device.newBuffer(bytes);
            if (normalBuf) normalBuf.write(0, bytes, normal.data());
        }

        oidn::FilterRef filter = device.newFilter("RT");
        filter.setImage("color", colorBuf, oidn::Format::Float3,
                        (size_t)width, (size_t)height);
        if (haveAlbedo && albedoBuf)
            filter.setImage("albedo", albedoBuf, oidn::Format::Float3,
                            (size_t)width, (size_t)height);
        if (haveNormal && normalBuf)
            filter.setImage("normal", normalBuf, oidn::Format::Float3,
                            (size_t)width, (size_t)height);
        filter.setImage("output", colorBuf, oidn::Format::Float3,
                        (size_t)width, (size_t)height);
        filter.set("hdr", true);
        filter.commit();
        filter.execute();

        // Copy the denoised result back into the caller's std::vector.
        // The same colorBuf was used for both the input and the output
        // image slots, so after execute() it holds the denoised pixels.
        colorBuf.read(0, bytes, color.data());

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
