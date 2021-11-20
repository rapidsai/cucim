/**
 * Code below is based on libtiff library which is under BSD-like license,
 * for providing lzw_decoder implementation.
 * The code is a port of the following file:
 *    https://gitlab.com/libtiff/libtiff/-/blob/8546f7ee994eacff0a563918096f16e0a6078fa2/libtiff/tif_lzw.c
 * , which is after v4.3.0.
 * Please see LICENSE-3rdparty.md for the detail.
 *
 * Changes
 * - Remove v5.0 specification compatibility
 * - Remove LZW_CHECKEOS which checks for strips w/o EOI code
 * - Remove encoder logic
 * - Remove 'register' keyword
 * - Handle unused variables/methods to avoid compiler errors
 **/


/****************************************************************************
 * Define missing types for libtiff's lzw decoder implementation
 ****************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cinttypes>

#include "lzw_libtiff.h"

#define _TIFFmalloc(...) malloc(__VA_ARGS__)
#define _TIFFmemset(...) memset(__VA_ARGS__)
#define _TIFFfree(...) free(__VA_ARGS__)
#define TIFFErrorExt(tif, module, ...)                                                                                 \
    {                                                                                                                  \
        (void)module;                                                                                                  \
    }                                                                                                                  \
    fprintf(stderr, __VA_ARGS__)

namespace cuslide::lzw
{

// **************************************************************************

/****************************************************************************
 * Define methods needed for libtiff's lzw decoder implementation
 ****************************************************************************/

// The following implementation is based on:
//   https://github.com/uclouvain/openjpeg/blob/37ac30ceff6640bbab502388c5e0fa0bff23f505/thirdparty/libtiff/tif_predict.c#L268

void horAcc8(uint8_t* cp0, tmsize_t cc, tmsize_t width_nbytes)
{
    unsigned char* cp = (unsigned char*)cp0;
    while (cc > 0)
    {
        tmsize_t remaining = width_nbytes;
        unsigned int cr = cp[0];
        unsigned int cg = cp[1];
        unsigned int cb = cp[2];
        remaining -= 3;
        cp += 3;
        while (remaining > 0)
        {
            cp[0] = (unsigned char)((cr += cp[0]) & 0xff);
            cp[1] = (unsigned char)((cg += cp[1]) & 0xff);
            cp[2] = (unsigned char)((cb += cp[2]) & 0xff);
            remaining -= 3;
            cp += 3;
        }
        cc -= width_nbytes;
    }
}

// **************************************************************************

/*
 * Copyright (c) 1988-1997 Sam Leffler
 * Copyright (c) 1991-1997 Silicon Graphics, Inc.
 *
 * Permission to use, copy, modify, distribute, and sell this software and
 * its documentation for any purpose is hereby granted without fee, provided
 * that (i) the above copyright notices and this permission notice appear in
 * all copies of the software and related documentation, and (ii) the names of
 * Sam Leffler and Silicon Graphics may not be used in any advertising or
 * publicity relating to the software without the specific, prior written
 * permission of Sam Leffler and Silicon Graphics.
 *
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * IN NO EVENT SHALL SAM LEFFLER OR SILICON GRAPHICS BE LIABLE FOR
 * ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 */

/*
 * TIFF Library.
 * Rev 5.0 Lempel-Ziv & Welch Compression Support
 *
 * This code is derived from the compress program whose code is
 * derived from software contributed to Berkeley by James A. Woods,
 * derived from original work by Spencer Thomas and Joseph Orost.
 *
 * The original Berkeley copyright notice appears below in its entirety.
 */
#include <stdio.h>

/*
 * Each strip of data is supposed to be terminated by a CODE_EOI.
 * If the following #define is included, the decoder will also
 * check for end-of-strip w/o seeing this code.  This makes the
 * library more robust, but also slower.
 */
#define LZW_CHECKEOS /* include checks for strips w/o EOI code */

#define MAXCODE(n) ((1L << (n)) - 1)
/*
 * The TIFF spec specifies that encoded bit
 * strings range from 9 to 12 bits.
 */
#define BITS_MIN 9 /* start with 9 bits */
#define BITS_MAX 12 /* max of 12 bit strings */
/* predefined codes */
#define CODE_CLEAR 256 /* code to clear string table */
#define CODE_EOI 257 /* end-of-information code */
#define CODE_FIRST 258 /* first free code entry */
#define CODE_MAX MAXCODE(BITS_MAX)
#define HSIZE 9001L /* 91% occupancy */
#define HSHIFT (13 - 8)
#define CSIZE (MAXCODE(BITS_MAX) + 1L)

/*
 * State block for each open TIFF file using LZW
 * compression/decompression.  Note that the predictor
 * state block must be first in this data structure.
 */
typedef struct
{
    unsigned short nbits; /* # of bits/code */
    unsigned short maxcode; /* maximum code for lzw_nbits */
    unsigned short free_ent; /* next free entry in hash table */
    unsigned long nextdata; /* next bits of i/o */
    long nextbits; /* # of valid bits in lzw_nextdata */

    int rw_mode; /* preserve rw_mode from init */
} LZWBaseState;

#define lzw_nbits base.nbits
#define lzw_maxcode base.maxcode
#define lzw_free_ent base.free_ent
#define lzw_nextdata base.nextdata
#define lzw_nextbits base.nextbits

/*
 * Encoding-specific state.
 */
typedef uint16_t hcode_t; /* codes fit in 16 bits */
typedef struct
{
    long hash;
    hcode_t code;
} hash_t;

/*
 * Decoding-specific state.
 */
typedef struct code_ent
{
    struct code_ent* next;
    unsigned short length; /* string len, including this token */
    unsigned char value; /* data value */
    unsigned char firstchar; /* first token of string */
} code_t;

typedef int (*decodeFunc)(TIFF*, uint8_t*, tmsize_t, uint16_t);

typedef struct
{
    LZWBaseState base;

    /* Decoding specific data */
    long dec_nbitsmask; /* lzw_nbits 1 bits, right adjusted */
    long dec_restart; /* restart count */
    decodeFunc dec_decode; /* regular or backwards compatible */
    code_t* dec_codep; /* current recognized code */
    code_t* dec_oldcodep; /* previously recognized code */
    code_t* dec_free_entp; /* next free entry */
    code_t* dec_maxcodep; /* max available entry */
    code_t* dec_codetab; /* kept separate for small machines */

    /* Encoding specific data */
    int enc_oldcode; /* last code encountered */
    long enc_checkpoint; /* point at which to clear table */
    long enc_ratio; /* current compression ratio */
    long enc_incount; /* (input) data bytes encoded */
    long enc_outcount; /* encoded (output) bytes */
    uint8_t* enc_rawlimit; /* bound on tif_rawdata buffer */
    hash_t* enc_hashtab; /* kept separate for small machines */
} LZWCodecState;

#define LZWState(tif) ((LZWBaseState*)(tif)->tif_data)
#define DecoderState(tif) ((LZWCodecState*)LZWState(tif))
#define EncoderState(tif) ((LZWCodecState*)LZWState(tif))

static int LZWDecode(TIFF* tif, uint8_t* op0, tmsize_t occ0, uint16_t s);

/*
 * LZW Decoder.
 */

#define NextCode(tif, sp, bp, code, get) get(sp, bp, code)

static int LZWSetupDecode(TIFF* tif)
{
    static const char module[] = "LZWSetupDecode";
    LZWCodecState* sp = DecoderState(tif);
    int code;

    if (sp == NULL)
    {
        /*
         * Allocate state block so tag methods have storage to record
         * values.
         */
        tif->tif_data = (uint8_t*)_TIFFmalloc(sizeof(LZWCodecState));
        if (tif->tif_data == NULL)
        {
            TIFFErrorExt(tif->tif_clientdata, module, "No space for LZW state block");
            return (0);
        }

        sp = DecoderState(tif);
        sp->dec_codetab = NULL;
        sp->dec_decode = NULL;
    }

    if (sp->dec_codetab == NULL)
    {
        sp->dec_codetab = (code_t*)_TIFFmalloc(CSIZE * sizeof(code_t));
        if (sp->dec_codetab == NULL)
        {
            TIFFErrorExt(tif->tif_clientdata, module, "No space for LZW code table");
            return (0);
        }
        /*
         * Pre-load the table.
         */
        code = 255;
        do
        {
            sp->dec_codetab[code].value = (unsigned char)code;
            sp->dec_codetab[code].firstchar = (unsigned char)code;
            sp->dec_codetab[code].length = 1;
            sp->dec_codetab[code].next = NULL;
        } while (code--);
        /*
         * Zero-out the unused entries
         */
        /* Silence false positive */
        /* coverity[overrun-buffer-arg] */
        _TIFFmemset(&sp->dec_codetab[CODE_CLEAR], 0, (CODE_FIRST - CODE_CLEAR) * sizeof(code_t));
    }
    return (1);
}

/*
 * Setup state for decoding a strip.
 */
static int LZWPreDecode(TIFF* tif, uint16_t s)
{
    static const char module[] = "LZWPreDecode";
    LZWCodecState* sp = DecoderState(tif);

    (void)s;
    assert(sp != NULL);
    if (sp->dec_codetab == NULL)
    {
        tif->tif_setupdecode(tif);
        if (sp->dec_codetab == NULL)
            return (0);
    }

    /*
     * Check for old bit-reversed codes.
     */
    if (tif->tif_rawcc >= 2 && tif->tif_rawdata[0] == 0 && (tif->tif_rawdata[1] & 0x1))
    {
        if (!sp->dec_decode)
        {
            TIFFErrorExt(tif->tif_clientdata, module, "Old-style LZW codes not supported");
            sp->dec_decode = LZWDecode;
        }
        return (0);
    }
    else
    {
        sp->lzw_maxcode = MAXCODE(BITS_MIN) - 1;
        sp->dec_decode = LZWDecode;
    }
    sp->lzw_nbits = BITS_MIN;
    sp->lzw_nextbits = 0;
    sp->lzw_nextdata = 0;

    sp->dec_restart = 0;
    sp->dec_nbitsmask = MAXCODE(BITS_MIN);
    sp->dec_free_entp = sp->dec_codetab + CODE_FIRST;
    /*
     * Zero entries that are not yet filled in.  We do
     * this to guard against bogus input data that causes
     * us to index into undefined entries.  If you can
     * come up with a way to safely bounds-check input codes
     * while decoding then you can remove this operation.
     */
    _TIFFmemset(sp->dec_free_entp, 0, (CSIZE - CODE_FIRST) * sizeof(code_t));
    sp->dec_oldcodep = &sp->dec_codetab[-1];
    sp->dec_maxcodep = &sp->dec_codetab[sp->dec_nbitsmask - 1];
    return (1);
}

/*
 * Decode a "hunk of data".
 */
#define GetNextCode(sp, bp, code)                                                                                      \
    {                                                                                                                  \
        nextdata = (nextdata << 8) | *(bp)++;                                                                          \
        nextbits += 8;                                                                                                 \
        if (nextbits < nbits)                                                                                          \
        {                                                                                                              \
            nextdata = (nextdata << 8) | *(bp)++;                                                                      \
            nextbits += 8;                                                                                             \
        }                                                                                                              \
        code = (hcode_t)((nextdata >> (nextbits - nbits)) & nbitsmask);                                                \
        nextbits -= nbits;                                                                                             \
    }

static void codeLoop(TIFF* tif, const char* module)
{
    TIFFErrorExt(tif->tif_clientdata, module, "Bogus encoding, loop in the code table; scanline %" PRIu32, tif->tif_row);
}

static int LZWDecode(TIFF* tif, uint8_t* op0, tmsize_t occ0, uint16_t s)
{
    static const char module[] = "LZWDecode";
    LZWCodecState* sp = DecoderState(tif);
    uint8_t* op = (uint8_t*)op0;
    long occ = (long)occ0;
    uint8_t* tp;
    uint8_t* bp;
    hcode_t code;
    int len;
    long nbits, nextbits, nbitsmask;
    unsigned long nextdata;
    code_t *codep, *free_entp, *maxcodep, *oldcodep;

    (void)s;
    assert(sp != NULL);
    assert(sp->dec_codetab != NULL);

    /*
      Fail if value does not fit in long.
    */
    if ((tmsize_t)occ != occ0)
        return (0);
    /*
     * Restart interrupted output operation.
     */
    if (sp->dec_restart)
    {
        long residue;

        codep = sp->dec_codep;
        residue = codep->length - sp->dec_restart;
        if (residue > occ)
        {
            /*
             * Residue from previous decode is sufficient
             * to satisfy decode request.  Skip to the
             * start of the decoded string, place decoded
             * values in the output buffer, and return.
             */
            sp->dec_restart += occ;
            do
            {
                codep = codep->next;
            } while (--residue > occ && codep);
            if (codep)
            {
                tp = op + occ;
                do
                {
                    *--tp = codep->value;
                    codep = codep->next;
                } while (--occ && codep);
            }
            return (1);
        }
        /*
         * Residue satisfies only part of the decode request.
         */
        op += residue;
        occ -= residue;
        tp = op;
        do
        {
            *--tp = codep->value;
            codep = codep->next;
        } while (--residue && codep);
        sp->dec_restart = 0;
    }

    bp = (uint8_t*)tif->tif_rawcp;
    nbits = sp->lzw_nbits;
    nextdata = sp->lzw_nextdata;
    nextbits = sp->lzw_nextbits;
    nbitsmask = sp->dec_nbitsmask;
    oldcodep = sp->dec_oldcodep;
    free_entp = sp->dec_free_entp;
    maxcodep = sp->dec_maxcodep;

    while (occ > 0)
    {
        NextCode(tif, sp, bp, code, GetNextCode);
        if (code == CODE_EOI)
            break;
        if (code == CODE_CLEAR)
        {
            do
            {
                free_entp = sp->dec_codetab + CODE_FIRST;
                _TIFFmemset(free_entp, 0, (CSIZE - CODE_FIRST) * sizeof(code_t));
                nbits = BITS_MIN;
                nbitsmask = MAXCODE(BITS_MIN);
                maxcodep = sp->dec_codetab + nbitsmask - 1;
                NextCode(tif, sp, bp, code, GetNextCode);
            } while (code == CODE_CLEAR); /* consecutive CODE_CLEAR codes */
            if (code == CODE_EOI)
                break;
            if (code > CODE_CLEAR)
            {
                TIFFErrorExt(tif->tif_clientdata, tif->tif_name, "LZWDecode: Corrupted LZW table at scanline %" PRIu32,
                             tif->tif_row);
                return (0);
            }
            *op++ = (uint8_t)code;
            occ--;
            oldcodep = sp->dec_codetab + code;
            continue;
        }
        codep = sp->dec_codetab + code;

        /*
         * Add the new entry to the code table.
         */
        if (free_entp < &sp->dec_codetab[0] || free_entp >= &sp->dec_codetab[CSIZE])
        {
            TIFFErrorExt(tif->tif_clientdata, module, "Corrupted LZW table at scanline %" PRIu32, tif->tif_row);
            return (0);
        }

        free_entp->next = oldcodep;
        if (free_entp->next < &sp->dec_codetab[0] || free_entp->next >= &sp->dec_codetab[CSIZE])
        {
            TIFFErrorExt(tif->tif_clientdata, module, "Corrupted LZW table at scanline %" PRIu32, tif->tif_row);
            return (0);
        }
        free_entp->firstchar = free_entp->next->firstchar;
        free_entp->length = free_entp->next->length + 1;
        free_entp->value = (codep < free_entp) ? codep->firstchar : free_entp->firstchar;
        if (++free_entp > maxcodep)
        {
            if (++nbits > BITS_MAX) /* should not happen */
                nbits = BITS_MAX;
            nbitsmask = MAXCODE(nbits);
            maxcodep = sp->dec_codetab + nbitsmask - 1;
        }
        oldcodep = codep;
        if (code >= 256)
        {
            /*
             * Code maps to a string, copy string
             * value to output (written in reverse).
             */
            if (codep->length == 0)
            {
                TIFFErrorExt(tif->tif_clientdata, module,
                             "Wrong length of decoded string: "
                             "data probably corrupted at scanline %" PRIu32,
                             tif->tif_row);
                return (0);
            }
            if (codep->length > occ)
            {
                /*
                 * String is too long for decode buffer,
                 * locate portion that will fit, copy to
                 * the decode buffer, and setup restart
                 * logic for the next decoding call.
                 */
                sp->dec_codep = codep;
                do
                {
                    codep = codep->next;
                } while (codep && codep->length > occ);
                if (codep)
                {
                    sp->dec_restart = (long)occ;
                    tp = op + occ;
                    do
                    {
                        *--tp = codep->value;
                        codep = codep->next;
                    } while (--occ && codep);
                    if (codep)
                        codeLoop(tif, module);
                }
                break;
            }
            len = codep->length;
            tp = op + len;
            do
            {
                *--tp = codep->value;
                codep = codep->next;
            } while (codep && tp > op);
            if (codep)
            {
                codeLoop(tif, module);
                break;
            }
            assert(occ >= len);
            op += len;
            occ -= len;
        }
        else
        {
            *op++ = (uint8_t)code;
            occ--;
        }
    }

    tif->tif_rawcc -= (tmsize_t)((uint8_t*)bp - tif->tif_rawcp);
    tif->tif_rawcp = (uint8_t*)bp;
    sp->lzw_nbits = (unsigned short)nbits;
    sp->lzw_nextdata = nextdata;
    sp->lzw_nextbits = nextbits;
    sp->dec_nbitsmask = nbitsmask;
    sp->dec_oldcodep = oldcodep;
    sp->dec_free_entp = free_entp;
    sp->dec_maxcodep = maxcodep;

    if (occ > 0)
    {
        TIFFErrorExt(tif->tif_clientdata, module, "Not enough data at scanline %" PRIu32 " (short %ld bytes)",
                     tif->tif_row, occ);
        return (0);
    }
    return (1);
}

static void LZWCleanup(TIFF* tif)
{
    assert(tif->tif_data != 0);

    if (DecoderState(tif)->dec_codetab)
        _TIFFfree(DecoderState(tif)->dec_codetab);

    if (EncoderState(tif)->enc_hashtab)
        _TIFFfree(EncoderState(tif)->enc_hashtab);

    _TIFFfree(tif->tif_data);
    tif->tif_data = NULL;
}

int TIFFInitLZW(TIFF* tif, int scheme)
{
    static const char module[] = "TIFFInitLZW";
    (void)scheme;
    assert(scheme == COMPRESSION_LZW);
    /*
     * Allocate state block so tag methods have storage to record values.
     */
    tif->tif_data = (uint8_t*)_TIFFmalloc(sizeof(LZWCodecState));
    if (tif->tif_data == NULL)
        goto bad;
    DecoderState(tif)->dec_codetab = NULL;
    DecoderState(tif)->dec_decode = NULL;
    EncoderState(tif)->enc_hashtab = NULL;

    /*
     * Install codec methods.
     */
    tif->tif_setupdecode = LZWSetupDecode;
    tif->tif_predecode = LZWPreDecode;
    tif->tif_decoderow = LZWDecode;
    tif->tif_decodestrip = LZWDecode;
    tif->tif_decodetile = LZWDecode;
    tif->tif_cleanup = LZWCleanup;
    return (1);
bad:
    TIFFErrorExt(tif->tif_clientdata, module, "No space for LZW state block");
    return (0);
}

/*
 * Copyright (c) 1985, 1986 The Regents of the University of California.
 * All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * James A. Woods, derived from original work by Spencer Thomas
 * and Joseph Orost.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the University of California, Berkeley.  The name of the
 * University may not be used to endorse or promote products derived
 * from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */

} // namespace cuslide::lzw