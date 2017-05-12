/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains configuration module implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-13
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include <windows.h>
#include <basetsd.h>
#include "configuration.h"
#include "argtable3.h"
#include "hc.h"

#define NUMBER_PARAM_FMT_STRING "%lu"
#define BIG_NUMBER_PARAM_FMT_STRING "%lli"

#define INVALID_DIGIT_PARAMETER "Invalid parameter --%s %s. Must be number" NEW_LINE

#define OPT_LIMIT_SHORT "z"
#define OPT_LIMIT_FULL "limit"
#define OPT_LIMIT_DESCR _("set the limit in bytes of the part of the file to calculate hash for. The whole file by default will be applied")

#define OPT_OFFSET_SHORT "q"
#define OPT_OFFSET_FULL "offset"
#define OPT_OFFSET_DESCR _("set start position within file to calculate hash from. Zero by default")

#define OPT_HELP_SHORT "h"
#define OPT_HELP_LONG "help"
#define OPT_HELP_DESCR _("print this help and exit")

#define OPT_TIME_SHORT "t"
#define OPT_TIME_LONG "time"
#define OPT_TIME_DESCR _("show calculation time (false by default)")

#define OPT_LOW_SHORT "l"
#define OPT_LOW_LONG "lower"
#define OPT_LOW_DESCR _("output hash using low case (false by default)")

#define OPT_VERIFY_SHORT "c"
#define OPT_VERIFY_LONG "checksumfile"
#define OPT_VERIFY_DESCR _("output hash in file checksum format")

#define OPT_SFV_LONG "sfv"
#define OPT_SFV_DESCR _("output hash in the SFV (Simple File Verification)  format (false by default). Only for CRC32.")

#define OPT_NOPROBE_LONG "noprobe"
#define OPT_NOPROBE_DESCR _("Disable hash crack time probing (how much time it may take)")

#define OPT_NOERR_LONG "noerroronfind"
#define OPT_NOERR_DESCR _("Disable error output while search files. False by default.")

#define OPT_THREAD_SHORT "T"
#define OPT_THREAD_LONG "threads"
#define OPT_THREAD_DESCR _("the number of threads to crack hash. The half of system processors by default. The value must be between 1 and processor count.")

#define OPT_SAVE_SHORT "o"
#define OPT_SAVE_LONG "save"
#define OPT_SAVE_DESCR _("save files' hashes into the file specified instead of console.")

#define OPT_HASH_DESCR _("hash algorithm. See all possible values below")

#define OPT_HASH_SHORT "m"
#define OPT_HASH_FULL "hash"

#define OPT_HASH_TYPE "<algorithm>"
#define OPT_CMD_TYPE "<command>"

#define OPT_BASE64_SHORT "b"
#define OPT_BASE64_FULL "base64"
#define OPT_BASE64_DESCR _("output hash as Base64")

#define STRING_CMD "string"
#define HASH_CMD "hash"
#define FILE_CMD "file"
#define DIR_CMD "dir"

#ifdef __STDC_WANT_SECURE_LIB__
#define SSCANF sscanf_s
#else
#define SSCANF sscanf
#endif

// Forwards
static uint32_t prconf_get_threads_count(struct arg_int* threads);
static BOOL prconf_read_offset_parameter(struct arg_str* offset, const char* option, apr_off_t* result);

static BOOL prconf_is_cmd(struct arg_str* cmd, const char* name) {
    return !strcmp(cmd->sval[0], name);
}

static BOOL prconf_is_string_cmd(struct arg_str* cmd) {
    return prconf_is_cmd(cmd, STRING_CMD);
}

static BOOL prconf_is_hash_cmd(struct arg_str* cmd) {
    return prconf_is_cmd(cmd, HASH_CMD);
}

static BOOL prconf_is_file_cmd(struct arg_str* cmd) {
    return prconf_is_cmd(cmd, FILE_CMD);
}

static BOOL prconf_is_dir_cmd(struct arg_str* cmd) {
    return prconf_is_cmd(cmd, DIR_CMD);
}

void conf_run_app(configuration_ctx_t* ctx) {
    int nerrorsS;
    int nerrorsH;
    int nerrorsF;
    int nerrorsD;

    // Only cmd mode
    struct arg_str* hashS = arg_str1(NULL, NULL, OPT_HASH_TYPE, OPT_HASH_DESCR);
    struct arg_str* hashH = arg_str1(NULL, NULL, OPT_HASH_TYPE, OPT_HASH_DESCR);
    struct arg_str* hashF = arg_str1(NULL, NULL, OPT_HASH_TYPE, OPT_HASH_DESCR);
    struct arg_str* hashD = arg_str1(NULL, NULL, OPT_HASH_TYPE, OPT_HASH_DESCR);

    struct arg_str* cmdS = arg_str1(NULL, NULL, OPT_CMD_TYPE, _("must be string"));
    struct arg_str* cmdH = arg_str1(NULL, NULL, OPT_CMD_TYPE, _("must be hash"));
    struct arg_str* cmdF = arg_str1(NULL, NULL, OPT_CMD_TYPE, _("must be file"));
    struct arg_str* cmdD = arg_str1(NULL, NULL, OPT_CMD_TYPE, _("must be dir"));

    struct arg_file* file = arg_file1("f", "file", NULL, _("full path to file to calculate hash sum of"));
    struct arg_str* dir = arg_str1("d", "dir", NULL, _("full path to dir to calculate all content's hashes"));
    struct arg_str* exclude = arg_str0("e", "exclude", NULL, _("exclude files that match the pattern specified. It's possible to use several patterns separated by ;"));
    struct arg_str* include = arg_str0("i", "include", NULL, _("include only files that match the pattern specified. It's possible to use several patterns separated by ;"));
    struct arg_str* string = arg_str1("s", "string", NULL, _("string to calculate hash sum for"));
    struct arg_str* digestH = arg_str0(OPT_HASH_SHORT, OPT_HASH_FULL, NULL, _("hash to restore initial string by"));
    struct arg_str* digestF = arg_str0(OPT_HASH_SHORT, OPT_HASH_FULL, NULL, _("hash to validate file"));
    struct arg_str* digestD = arg_str0(OPT_HASH_SHORT, OPT_HASH_FULL, NULL, _("hash to validate files in directory"));
    struct arg_lit* base64digest = arg_lit0("b", "base64hash", _("interpret hash as Base64"));
    struct arg_lit* output_in_base64_s = arg_lit0(OPT_BASE64_SHORT, OPT_BASE64_FULL, OPT_BASE64_DESCR);
    struct arg_lit* output_in_base64_f = arg_lit0(OPT_BASE64_SHORT, OPT_BASE64_FULL, OPT_BASE64_DESCR);
    struct arg_lit* output_in_base64_d = arg_lit0(OPT_BASE64_SHORT, OPT_BASE64_FULL, OPT_BASE64_DESCR);
    struct arg_str* dict = arg_str0("a",
                                    "dict",
                                    NULL,
                                    _("initial string's dictionary. All digits, upper and lower case latin symbols by default"));
    struct arg_int* min = arg_int0("n", "min", NULL, _("set minimum length of the string to restore. 1 by default"));
    struct arg_int* max = arg_int0("x",
                                   "max",
                                   NULL,
                                   _("set maximum length of the string to restore. 10 by default"));
    struct arg_str* limitF = arg_str0(OPT_LIMIT_SHORT, OPT_LIMIT_FULL, "<number>", OPT_LIMIT_DESCR); // -V656
    struct arg_str* limitD = arg_str0(OPT_LIMIT_SHORT, OPT_LIMIT_FULL, "<number>", OPT_LIMIT_DESCR); // -V656
    struct arg_str* offsetF = arg_str0(OPT_OFFSET_SHORT, OPT_OFFSET_FULL, "<number>", OPT_OFFSET_DESCR); // -V656
    struct arg_str* offsetD = arg_str0(OPT_OFFSET_SHORT, OPT_OFFSET_FULL, "<number>", OPT_OFFSET_DESCR); // -V656
    struct arg_str* search = arg_str0("H", "search", NULL, _("hash to search a file that matches it"));

    struct arg_lit* recursively = arg_lit0("r", "recursively", _("scan directory recursively"));
    struct arg_lit* performance = arg_lit0("p", "performance", _("test performance by cracking 12345 string hash"));

    // Common options
    struct arg_lit* helpS = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_lit* helpH = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_lit* helpF = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_lit* helpD = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_lit* timeF = arg_lit0(OPT_TIME_SHORT, OPT_TIME_LONG, OPT_TIME_DESCR); // -V656
    struct arg_lit* timeD = arg_lit0(OPT_TIME_SHORT, OPT_TIME_LONG, OPT_TIME_DESCR); // -V656
    struct arg_lit* lowerS = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* lowerH = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* lowerF = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* lowerD = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* verifyF = arg_lit0(OPT_VERIFY_SHORT, OPT_VERIFY_LONG, OPT_VERIFY_DESCR); // -V656
    struct arg_lit* verifyD = arg_lit0(OPT_VERIFY_SHORT, OPT_VERIFY_LONG, OPT_VERIFY_DESCR); // -V656
    struct arg_lit* noProbe = arg_lit0(NULL, OPT_NOPROBE_LONG, OPT_NOPROBE_DESCR);
    struct arg_lit* noErrorOnFind = arg_lit0(NULL, OPT_NOERR_LONG, OPT_NOERR_DESCR);
    struct arg_int* threads = arg_int0(OPT_THREAD_SHORT, OPT_THREAD_LONG, NULL, OPT_THREAD_DESCR);
    struct arg_file* saveF = arg_file0(OPT_SAVE_SHORT, OPT_SAVE_LONG, NULL, OPT_SAVE_DESCR);
    struct arg_file* saveD = arg_file0(OPT_SAVE_SHORT, OPT_SAVE_LONG, NULL, OPT_SAVE_DESCR);
    struct arg_lit* sfvF = arg_lit0(NULL, OPT_SFV_LONG, OPT_SFV_DESCR);
    struct arg_lit* sfvD = arg_lit0(NULL, OPT_SFV_LONG, OPT_SFV_DESCR);

    struct arg_end* endS = arg_end(10);
    struct arg_end* endH = arg_end(10);
    struct arg_end* endF = arg_end(10);
    struct arg_end* endD = arg_end(10);

    // Command line mode table
    void* argtableS[] = { hashS, cmdS, string, output_in_base64_s, lowerS, helpS, endS };
    void* argtableH[] = { hashH, cmdH, digestH, base64digest, dict, min, max, performance, noProbe, threads, lowerH, helpH, endH };
    void* argtableF[] = { hashF, cmdF, file, digestF, limitF, offsetF, verifyF, saveF, timeF, sfvF, lowerF, output_in_base64_f, helpF, endF };
    void* argtableD[] = { hashD, cmdD, dir, digestD, exclude, include, limitD, offsetD, search, recursively, verifyD, saveD, timeD, sfvD, lowerD, output_in_base64_d, noErrorOnFind, helpD, endD };

    if(arg_nullcheck(argtableS) != 0 && arg_nullcheck(argtableH) != 0 && arg_nullcheck(argtableF) != 0 && arg_nullcheck(argtableD) != 0) {
        hc_print_syntax(argtableS, argtableH, argtableF, argtableD);
        goto cleanup;
    }

    nerrorsS = arg_parse(ctx->argc, ctx->argv, argtableS);
    nerrorsH = arg_parse(ctx->argc, ctx->argv, argtableH);
    nerrorsF = arg_parse(ctx->argc, ctx->argv, argtableF);
    nerrorsD = arg_parse(ctx->argc, ctx->argv, argtableD);

    if(helpS->count > 0 || ctx->argc == 1) {
        hc_print_syntax(argtableS, argtableH, argtableF, argtableD);
        goto cleanup;
    }

    if(ctx->argc > 1 && !prconf_is_string_cmd(cmdS) && !prconf_is_hash_cmd(cmdS) && !prconf_is_file_cmd(cmdS) && !prconf_is_dir_cmd(cmdS)) {
        lib_printf(_("Invalid command one of: %s, %s, %s or %s expected"), STRING_CMD, HASH_CMD, FILE_CMD, DIR_CMD);
        goto cleanup;
    }

    if(prconf_is_string_cmd(cmdS) && nerrorsS) {
        hc_print_cmd_syntax(argtableS, endS);
        goto cleanup;
    }

    if(prconf_is_hash_cmd(cmdH) && nerrorsH) {
        hc_print_cmd_syntax(argtableH, endH);
        goto cleanup;
    }

    if(prconf_is_file_cmd(cmdF) && nerrorsF) {
        hc_print_cmd_syntax(argtableF, endF);
        goto cleanup;
    }

    if(prconf_is_dir_cmd(cmdD) && nerrorsD) {
        hc_print_cmd_syntax(argtableD, endD);
        goto cleanup;
    }

    builtin_ctx_t* builtin_ctx = apr_pcalloc(ctx->pool, sizeof(builtin_ctx_t));
    builtin_ctx->is_print_low_case_ = lowerS->count;
    builtin_ctx->hash_algorithm_ = hashS->sval[0];
    builtin_ctx->pfn_output_ = out_output_to_console;

    // run string builtin
    if(nerrorsS == 0) {
        string_builtin_ctx_t* str_ctx = apr_pcalloc(ctx->pool, sizeof(string_builtin_ctx_t));
        str_ctx->builtin_ctx_ = builtin_ctx;
        str_ctx->string_ = string->sval[0];
        str_ctx->is_base64_ = output_in_base64_s->count;

        ctx->pfn_on_string(builtin_ctx, str_ctx, ctx->pool);

        goto cleanup;
    }

    // run hash builtin
    if(nerrorsH == 0) {
        hash_builtin_ctx_t* hash_ctx = apr_pcalloc(ctx->pool, sizeof(hash_builtin_ctx_t));
        hash_ctx->builtin_ctx_ = builtin_ctx;
        hash_ctx->hash_ = digestH->sval[0];
        hash_ctx->is_base64_ = base64digest->count;
        hash_ctx->no_probe_ = noProbe->count;
        hash_ctx->performance_ = performance->count;
        hash_ctx->threads_ = prconf_get_threads_count(threads);

        if(dict->count > 0) {
            hash_ctx->dictionary_ = dict->sval[0];
        }
        if(min->count > 0) {
            hash_ctx->min_ = min->ival[0];
        }
        if(max->count > 0) {
            hash_ctx->max_ = max->ival[0];
        }

        ctx->pfn_on_hash(builtin_ctx, hash_ctx, ctx->pool);

        goto cleanup;
    }

    apr_off_t limit_value = 0;
    apr_off_t offset_value = 0;

    if(!prconf_read_offset_parameter(limitF, OPT_LIMIT_FULL, &limit_value)) {
        goto cleanup;
    }

    if(!prconf_read_offset_parameter(offsetF, OPT_OFFSET_FULL, &offset_value)) {
        goto cleanup;
    }

    // run file builtin
    if(nerrorsF == 0) {
        file_builtin_ctx_t* file_ctx = apr_palloc(ctx->pool, sizeof(file_builtin_ctx_t));
        file_ctx->builtin_ctx_ = builtin_ctx;
        file_ctx->file_path_ = file->filename[0];
        file_ctx->limit_ = limit_value ? limit_value : MAXLONG64;
        file_ctx->offset_ = offset_value;
        file_ctx->show_time_ = timeF->count;
        file_ctx->is_verify_ = verifyF->count;
        file_ctx->result_in_sfv_ = sfvF->count;

        file_ctx->hash_ = !digestF->count ? NULL : digestF->sval[0];
        file_ctx->save_result_path_ = !saveF->count ? NULL : saveF->filename[0];

        ctx->pfn_on_file(builtin_ctx, file_ctx, ctx->pool);

        goto cleanup;
    }

    if(nerrorsD == 0) {
        dir_builtin_ctx_t* dir_ctx = apr_palloc(ctx->pool, sizeof(dir_builtin_ctx_t));
        dir_ctx->builtin_ctx_ = builtin_ctx;
        dir_ctx->dir_path_ = dir->sval[0];
        dir_ctx->limit_ = limit_value ? limit_value : MAXLONG64;
        dir_ctx->offset_ = offset_value;
        dir_ctx->show_time_ = timeD->count;
        dir_ctx->is_verify_ = verifyD->count;
        dir_ctx->result_in_sfv_ = sfvD->count;
        dir_ctx->no_error_on_find_ = noErrorOnFind->count;
        dir_ctx->recursively_ = recursively->count;
        dir_ctx->include_pattern_ = include->count > 0 ? include->sval[0] : NULL;
        dir_ctx->exclude_pattern_ = exclude->count > 0 ? exclude->sval[0] : NULL;
        dir_ctx->hash_ = !digestD->count ? NULL : digestD->sval[0];
        dir_ctx->search_hash_ = search->count > 0 ? search->sval[0] : NULL;
        dir_ctx->save_result_path_ = !saveD->count ? NULL : saveD->filename[0];

        ctx->pfn_on_dir(builtin_ctx, dir_ctx, ctx->pool);
    }

cleanup:
    /* deallocate each non-null entry in argtables */
    arg_freetable(argtableS, sizeof argtableS / sizeof argtableS[0]);
    arg_freetable(argtableH, sizeof argtableH / sizeof argtableH[0]);
    arg_freetable(argtableF, sizeof argtableF / sizeof argtableF[0]);
    arg_freetable(argtableD, sizeof argtableD / sizeof argtableD[0]);
}

uint32_t prconf_get_threads_count(struct arg_int* threads) {
    uint32_t num_of_threads;
    uint32_t processors = lib_get_processor_count();

    if(threads->count > 0) {
        num_of_threads = (uint32_t)threads->ival[0];
    } else {
        num_of_threads = processors == 1 ? 1 : MIN(processors, processors / 2);
    }
    if(num_of_threads < 1 || num_of_threads > processors) {
        uint32_t def = processors == 1 ? processors : processors / 2;
        lib_printf(_("Threads number must be between 1 and %u but it was set to %lu. Reset to default %u"), processors, num_of_threads, def);
        lib_new_line();
        num_of_threads = def;
    }
    return num_of_threads;
}

BOOL prconf_read_offset_parameter(struct arg_str* offset, const char* option, apr_off_t* result) {
    if(offset->count > 0) {
        if(!SSCANF(offset->sval[0], BIG_NUMBER_PARAM_FMT_STRING, result)) {
            lib_printf(INVALID_DIGIT_PARAMETER, option, offset->sval[0]);
            return FALSE;
        }

        if(*result < 0) {
            hc_print_copyright();
            lib_printf(_("Invalid %s option must be positive but was %lli"), option, *result);
            lib_new_line();
            return FALSE;
        }
    }
    return TRUE;
}
