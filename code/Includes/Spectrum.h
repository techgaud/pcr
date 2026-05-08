#pragma once

#include <algorithm>
#include <array>
#include <cmath>

// Sampled visible-light spectrum, 400-700 nm at 5 nm intervals.
//
// 61 samples is the standard "physically-based rendering for a hobby
// project" resolution. Mitsuba and PBRT ship with similar sampling
// (PBRT v4 uses 360-830 nm at 1 nm = 471 samples, but most of that
// range is invisible to humans and adds cost without changing output).
// 5 nm resolution captures every meaningful feature of the CIE 1931
// observer functions and most natural illuminant SPDs.
//
// Storage is ~244 bytes per Spectrum. Cheap enough to copy by value
// for a hobby renderer; expensive enough that you don't want to
// allocate one per ray. The plan when this lands in the path tracer
// is to track a single (lambda, radiance) scalar pair per ray, not a
// full Spectrum, so the heavy storage only lives on materials and
// lights.
//
// All operations are header-only and constexpr-friendly so the
// compiler can fold the common cases (scalar multiply, add, etc).
class Spectrum
{
public:
    static constexpr float kLambdaMin = 400.f;
    static constexpr float kLambdaMax = 700.f;
    static constexpr int   kSamples   = 61;
    static constexpr float kStep      = (kLambdaMax - kLambdaMin) / (kSamples - 1);

    Spectrum() : _samples{} {}

    explicit Spectrum(float c)
    {
        _samples.fill(c);
    }

    explicit Spectrum(const std::array<float, kSamples> &s) : _samples{s} {}

    // Build a spectrum from Jakob 2019 sigmoid-polynomial coefficients.
    // S(lambda) = sigmoid(c0 + c1*lambda' + c2*lambda'^2) where
    // lambda' is the wavelength normalized to [-1, 1] across the
    // visible range. Sigmoid is the smooth saturator
    // S(x) = 0.5 + x / (2 sqrt(1 + x^2)) which guarantees output in
    // [0, 1] for any coefficient triple, automatically enforcing
    // physical reflectance bounds.
    //
    // Used by RGBToSpectrum::fitSpectrum at scene-load time to
    // upsample RGB albedos and emissions into spectra. The
    // coefficients are computed once per material; the resulting
    // Spectrum is what materials store and what the path tracer
    // samples at runtime.
    static Spectrum fromSigmoidCoefficients(float c0, float c1, float c2)
    {
        constexpr float kLambdaMid   = 0.5f * (kLambdaMin + kLambdaMax);
        constexpr float kLambdaHalf  = 0.5f * (kLambdaMax - kLambdaMin);
        Spectrum s;
        for (int i = 0; i < kSamples; i++)
        {
            float lambdaNorm = (lambdaAt(i) - kLambdaMid) / kLambdaHalf;
            float p = c0 + lambdaNorm * (c1 + lambdaNorm * c2);
            s._samples[i] = 0.5f + p / (2.f * std::sqrt(1.f + p * p));
        }
        return s;
    }

    // Wavelength of the i-th sample. Convenience for table lookups
    // when building an SPD by hand or comparing against published
    // spectral data.
    static constexpr float lambdaAt(int i)
    {
        return kLambdaMin + i * kStep;
    }

    // Look up radiance at an arbitrary wavelength via linear
    // interpolation between the two surrounding samples. Returns 0
    // outside [kLambdaMin, kLambdaMax]; the path tracer should never
    // sample those wavelengths in the first place but the bounds
    // check keeps integrate-style accumulators honest.
    float operator()(float lambda) const
    {
        if (lambda < kLambdaMin || lambda > kLambdaMax) return 0.f;
        float t = (lambda - kLambdaMin) / kStep;
        int   i = (int)t;
        if (i >= kSamples - 1) return _samples[kSamples - 1];
        float f = t - i;
        return _samples[i] * (1.f - f) + _samples[i + 1] * f;
    }

    // Direct sample access for code paths that already have an index
    // (e.g. the integrate-against-CIE loops in CIE.h).
    float &operator[](int i) { return _samples[i]; }
    float  operator[](int i) const { return _samples[i]; }

    // Componentwise arithmetic. * and *= are the workhorse for BSDF
    // evaluation (radiance times reflectance per wavelength); + is
    // for accumulating samples in an estimator.
    Spectrum &operator+=(const Spectrum &o) { for (int i = 0; i < kSamples; i++) _samples[i] += o._samples[i]; return *this; }
    Spectrum &operator-=(const Spectrum &o) { for (int i = 0; i < kSamples; i++) _samples[i] -= o._samples[i]; return *this; }
    Spectrum &operator*=(const Spectrum &o) { for (int i = 0; i < kSamples; i++) _samples[i] *= o._samples[i]; return *this; }
    Spectrum &operator/=(const Spectrum &o) { for (int i = 0; i < kSamples; i++) _samples[i] /= o._samples[i]; return *this; }
    Spectrum &operator*=(float s)           { for (int i = 0; i < kSamples; i++) _samples[i] *= s; return *this; }
    Spectrum &operator/=(float s)           { float inv = 1.f / s; return *this *= inv; }

    Spectrum operator+(const Spectrum &o) const { Spectrum r = *this; r += o; return r; }
    Spectrum operator-(const Spectrum &o) const { Spectrum r = *this; r -= o; return r; }
    Spectrum operator*(const Spectrum &o) const { Spectrum r = *this; r *= o; return r; }
    Spectrum operator/(const Spectrum &o) const { Spectrum r = *this; r /= o; return r; }
    Spectrum operator*(float s)           const { Spectrum r = *this; r *= s; return r; }
    Spectrum operator/(float s)           const { Spectrum r = *this; r /= s; return r; }

    // Scalar-on-the-left multiply for natural BSDF math (e.g. cos *
    // spectrum). Free function to support 0.5f * spec syntax.
    friend Spectrum operator*(float s, const Spectrum &spec) { return spec * s; }

    // Energy summary. Sum of all samples scaled by step width is a
    // discrete approximation of integral over wavelength; useful for
    // tone-map exposure and for assertions like "albedo integrates
    // to <= max width" (energy conservation).
    float integrate() const
    {
        float acc = 0.f;
        for (int i = 0; i < kSamples; i++) acc += _samples[i];
        return acc * kStep;
    }

    // True if every sample is exactly zero. Path tracer can early-
    // out on zero-throughput rays with this.
    bool isBlack() const
    {
        for (int i = 0; i < kSamples; i++) if (_samples[i] != 0.f) return false;
        return true;
    }

    // Clamp negative values to zero. Spectral reflectance and
    // emission are physically non-negative, but RGB-to-spectrum
    // upsampling (Jakob 2019, phase 3) can produce small negative
    // overshoots near saturation that we want to clip.
    Spectrum clampNonNegative() const
    {
        Spectrum r;
        for (int i = 0; i < kSamples; i++) r._samples[i] = std::max(0.f, _samples[i]);
        return r;
    }

private:
    std::array<float, kSamples> _samples;
};
