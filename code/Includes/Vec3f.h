#pragma once

#include <array>
#include <cmath>

struct Vec3f
{
public:
    Vec3f() : _values{0, 0, 0} {}
    Vec3f(float a, float b, float c) : _values{a, b, c} {}

private:
    std::array<float, 3> _values;

public:
    inline float dot(const Vec3f &other) const
    {
        return _values[0] * other[0] + _values[1] * other[1] + _values[2] * other[2];
    }

    inline Vec3f cross(const Vec3f &b) const
    {
        Vec3f out;
        out[0] = _values[1] * b[2] - _values[2] * b[1];
        out[1] = _values[2] * b[0] - _values[0] * b[2];
        out[2] = _values[0] * b[1] - _values[1] * b[0];
        return out;
    }

    inline float length()
    {
        return std::sqrt(this->dot(*this));
    }

    inline Vec3f normalize()
    {
        auto mag = std::sqrt(this->dot(*this));
        return {_values[0] / mag, _values[1] / mag, _values[2] / mag};
    }

    inline Vec3f reflect(const Vec3f &N) const
    {
        auto I = *this;
        return I - (N * I.dot(N) * 2.f);
    }

    inline float &operator[](int i)
    {
        return _values[i];
    }

    inline float operator[](int i) const
    {
        return _values[i];
    }

    inline Vec3f operator-(const Vec3f &other) const
    {
        return {_values[0] - other[0], _values[1] - other[1], _values[2] - other[2]};
    }

    inline Vec3f operator+(const Vec3f &other) const
    {
        return {_values[0] + other[0], _values[1] + other[1], _values[2] + other[2]};
    }

    inline Vec3f &operator+(const Vec3f &other)
    {
        _values[0] += other[0];
        _values[1] += other[1];
        _values[2] += other[2];
        return *this;
    }

    inline Vec3f &operator+=(const Vec3f &other)
    {
        return *this + other;
    }

    inline Vec3f operator*(float scale) const
    {
        return {_values[0] * scale, _values[1] * scale, _values[2] * scale};
    }

    inline Vec3f operator*(Vec3f &other) const
    {
        return {_values[0] * other[0], _values[1] * other[1], _values[2] * other[2]};
    }

    inline Vec3f &operator*=(float scale)
    {
        _values[0] *= scale;
        _values[1] *= scale;
        _values[2] *= scale;
        return *this;
    }

    inline Vec3f &operator/=(float scale)
    {
        _values[0] /= scale;
        _values[1] /= scale;
        _values[2] /= scale;
        return *this;
    }

    inline Vec3f operator/(float scale) const
    {
        return {_values[0] / scale, _values[1] / scale, _values[2] / scale};
    }
};