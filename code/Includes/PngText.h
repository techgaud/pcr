#pragma once

#include <string>

#include "lodepng.h"

// Append a tEXt chunk to a LodePNGInfo's "between PLTE and IDAT" unknown-
// chunks slot, so the encoder writes it BEFORE IDAT instead of after.
//
// Why bypass lodepng_add_text:
//   lodepng's encoder hardcodes the tEXt/zTXt write block to run AFTER
//   the IDAT chunk (see lodepng.cpp::lodepng_encode flow). PNG readers
//   accept text chunks at either position, but tools like ExifTool warn
//   when they appear after IDAT ("Text/EXIF chunk(s) found after PNG
//   IDAT"). Putting them before IDAT silences the warning and matches
//   the convention every other tooling pipeline expects.
//
//   lodepng's "unknown chunks" mechanism is intended for chunk types
//   it doesn't natively support, but it works fine for known types
//   too as long as we don't ALSO call lodepng_add_text with the same
//   key (that would emit duplicates: one before IDAT from us, one
//   after from lodepng's tEXt block). We don't, so we're safe.
//
//   tEXt chunk body format per the PNG spec:
//     keyword (1-79 Latin-1 bytes) + 0x00 + text (Latin-1, no null)
//   lodepng_chunk_create handles the chunk header, length, and CRC.
//
// On allocation failure or invalid keyword length, returns the lodepng
// error code; callers in this project ignore non-zero returns the same
// way they ignore lodepng_add_text returns (these are fatal-only paths
// that haven't tripped in practice).
inline unsigned pngAddTextBeforeIdat(LodePNGInfo *info,
                                     const char *keyword,
                                     const char *text)
{
    size_t kLen = 0;
    while (keyword[kLen] != '\0') kLen++;
    size_t tLen = 0;
    while (text[tLen] != '\0') tLen++;

    // PNG spec caps keyword at 79 bytes. Match lodepng's own validation.
    if (kLen < 1 || kLen > 79) return 67;

    std::string body;
    body.reserve(kLen + 1 + tLen);
    body.append(keyword, kLen);
    body.push_back('\0');
    body.append(text, tLen);

    return lodepng_chunk_create(
        &info->unknown_chunks_data[1],
        &info->unknown_chunks_size[1],
        body.size(),
        "tEXt",
        reinterpret_cast<const unsigned char *>(body.data()));
}

inline unsigned pngAddTextBeforeIdat(LodePNGInfo *info,
                                     const char *keyword,
                                     const std::string &text)
{
    return pngAddTextBeforeIdat(info, keyword, text.c_str());
}
