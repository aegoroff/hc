/*!
 * \brief   The file contains Hash LINQ implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#include <stdio.h>
#include <locale.h>

#include "targetver.h"
#include "argtable2.h"
#include "encoding.h"
#include "hc.h"

#include "../srclib/bf.h"
#include "compiler.h"
#include "../linq2hash/hashes.h"
#ifdef WIN32
#include "../srclib/dbg_helpers.h"
#endif

#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND
#define LINE_FEED '\n'

#define PATH_ELT_SEPARATOR '\\'
#define NUMBER_PARAM_FMT_STRING "%lu"
#define BIG_NUMBER_PARAM_FMT_STRING "%llu"

#define INVALID_DIGIT_PARAMETER "Invalid parameter --%s %s. Must be number" NEW_LINE
#define INCOMPATIBLE_OPTIONS_HEAD "Incompatible options: "

#define OPT_LIMIT_SHORT "z"
#define OPT_LIMIT_FULL "limit"
#define OPT_LIMIT_DESCR "set the limit in bytes of the part of the file to calculate hash for. The whole file by default will be applied"

#define OPT_OFFSET_SHORT "q"
#define OPT_OFFSET_FULL "offset"
#define OPT_OFFSET_DESCR "set start position within file to calculate hash from. Zero by default"

#define PATTERN_MATCH_DESCR_TAIL "the pattern specified. It's possible to use several patterns separated by ;"
#define MAX_DEFAULT_STR "10"

#define MAX_LINE_SIZE 32 * BINARY_THOUSAND - 1

static char* alphabet = DIGITS LOW_CASE UPPER_CASE;

#define PROG_EXE PROGRAM_NAME ".exe"

#define OPT_HELP_SHORT "h"
#define OPT_HELP_LONG "help"
#define OPT_HELP_DESCR "print this help and exit"

#define OPT_TIME_SHORT "t"
#define OPT_TIME_LONG "time"
#define OPT_TIME_DESCR "show calculation time (false by default)"

#define OPT_LOW_SHORT "l"
#define OPT_LOW_LONG "lower"
#define OPT_LOW_DESCR "output hash using low case (false by default)"

#define OPT_VERIFY_LONG "checksumfile"
#define OPT_VERIFY_DESCR "output hash in file checksum format"

#define OPT_SFV_LONG "sfv"
#define OPT_SFV_DESCR "output hash in the SFV (Simple File Verification)  format (false by default). Only for CRC32."

#define OPT_NOPROBE_LONG "noprobe"
#define OPT_NOPROBE_DESCR "Disable hash crack time probing (how much time it may take)"

#define OPT_NOERR_LONG "noerroronfind"
#define OPT_NOERR_DESCR "Disable error output while search files. False by default."

#define OPT_THREAD_SHORT "T"
#define OPT_THREAD_LONG "threads"
#define OPT_THREAD_DESCR "the number of threads to crack hash. The half of system processors by default. The value must be between 1 and processor count."

#define OPT_SAVE_SHORT "o"
#define OPT_SAVE_LONG "save"
#define OPT_SAVE_DESCR "save files' hashes into the file specified instead of console."

#define OPT_PARAM_SHORT "P"
#define OPT_PARAM_LONG "param"
#define OPT_PARAM_DESCR "path to file that will be validated using one or more queries"

#define OPT_SYNT_SHORT "S"
#define OPT_SYNT_LONG "syntaxonly"
#define OPT_SYNT_DESCR "only validate syntax. Do not run actions"

#define OPT_C_SHORT "C"
#define OPT_C_LONG "command"
#define OPT_C_DESCR "query text from command line"

#define OPT_F_SHORT "F"
#define OPT_F_LONG "query"
#define OPT_F_DESCR "one or more query files"

#define OPT_HASH_DESCR "hash algorithm. See all possible values below"
#define OPT_MODE_DESCR "mode."

// Forwards
void prhc_query_from_command_line(const char* cmd, const char* param, program_options_t* options, apr_pool_t* pool);

void prhc_query_from_files(struct arg_file* files, const char* param, program_options_t* options, apr_pool_t* pool);

uint32_t prhc_get_threads_count(struct arg_int* threads);

void prhc_main_command_line(
    const char* algorithm,
    struct arg_str* string,
    struct arg_lit* performance,
    struct arg_str* digest,
    struct arg_str* base64digest,
    struct arg_file* file,
    struct arg_str* dir,
    struct arg_str* include,
    struct arg_str* exclude,
    struct arg_str* search,
    struct arg_str* dict,
    struct arg_int* min,
    struct arg_int* max,
    struct arg_str* limit,
    struct arg_str* offset,
    struct arg_lit* recursively,
    program_options_t* options,
    apr_pool_t* pool);

void hc_print_syntax(void* argtableS[], void* argtableH[], void* argtableF[], void* argtableD[]);
void hc_print_table_syntax(void* argtable);

int main(int argc, const char* const argv[]) {
    apr_pool_t* pool = NULL;
    apr_status_t status = APR_SUCCESS;
    program_options_t* options = NULL;
    int nerrors;
    int nerrorsS;
    int nerrorsH;
    int nerrorsF;
    int nerrorsD;

    // Only cmd mode
    struct arg_str* hashS = arg_str1(NULL, NULL, NULL, OPT_HASH_DESCR);
    struct arg_str* hashH = arg_str1(NULL, NULL, NULL, OPT_HASH_DESCR);
    struct arg_str* hashF = arg_str1(NULL, NULL, NULL, OPT_HASH_DESCR);
    struct arg_str* hashD = arg_str1(NULL, NULL, NULL, OPT_HASH_DESCR);

    struct arg_str* modeS = arg_str1(NULL, NULL, NULL, OPT_MODE_DESCR "must be string");
    struct arg_str* modeH = arg_str1(NULL, NULL, NULL, OPT_MODE_DESCR "must be hash");
    struct arg_str* modeF = arg_str1(NULL, NULL, NULL, OPT_MODE_DESCR "must be file");
    struct arg_str* modeD = arg_str1(NULL, NULL, NULL, OPT_MODE_DESCR "must be dir");

    struct arg_file* file = arg_file0("f", "file", NULL, "full path to file to calculate hash sum of");
    struct arg_str* dir = arg_str0("d", "dir", NULL, "full path to dir to calculate all content's hashes");
    struct arg_str* exclude = arg_str0("e", "exclude", NULL, "exclude files that match " PATTERN_MATCH_DESCR_TAIL);
    struct arg_str* include = arg_str0("i", "include", NULL, "include only files that match " PATTERN_MATCH_DESCR_TAIL);
    struct arg_str* string = arg_str0("s", "string", NULL, "string to calculate hash sum for");
    struct arg_str* digest = arg_str0("m", "hash", NULL, "hash to validate file or to find initial string (crack)");
    struct arg_str* base64digest = arg_str0("b", "base64hash", NULL, "like -m(--hash) option but hash in Base64 form.");
    struct arg_str* dict = arg_str0("a",
                                    "dict",
                                    NULL,
                                    "initial string's dictionary. All digits, upper and lower case latin symbols by default");
    struct arg_int* min = arg_int0("n", "min", NULL, "set minimum length of the string to restore using option crack (c). 1 by default");
    struct arg_int* max = arg_int0("x",
                                   "max",
                                   NULL,
                                   "set maximum length of the string to restore  using option crack (c). " MAX_DEFAULT_STR " by default");
    struct arg_str* limitF = arg_str0(OPT_LIMIT_SHORT, OPT_LIMIT_FULL, "<number>", OPT_LIMIT_DESCR);
    struct arg_str* limitD = arg_str0(OPT_LIMIT_SHORT, OPT_LIMIT_FULL, "<number>", OPT_LIMIT_DESCR);
    struct arg_str* offsetF = arg_str0(OPT_OFFSET_SHORT, OPT_OFFSET_FULL, "<number>", OPT_OFFSET_DESCR);
    struct arg_str* offsetD = arg_str0(OPT_OFFSET_SHORT, OPT_OFFSET_FULL, "<number>", OPT_OFFSET_DESCR);
    struct arg_str* search = arg_str0("H", "search", NULL, "hash to search a file that matches it");

    struct arg_lit* recursively = arg_lit0("r", "recursively", "scan directory recursively");
    struct arg_lit* performance = arg_lit0("p", "performance", "test performance by cracking 123 string hash");


    // Common options
    struct arg_lit* helpS = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_lit* helpH = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_lit* helpF = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_lit* helpD = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_lit* timeF = arg_lit0(OPT_TIME_SHORT, OPT_TIME_LONG, OPT_TIME_DESCR);
    struct arg_lit* timeD = arg_lit0(OPT_TIME_SHORT, OPT_TIME_LONG, OPT_TIME_DESCR);
    struct arg_lit* lowerS = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* lowerH = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* lowerF = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* lowerD = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* verifyF = arg_lit0(NULL, OPT_VERIFY_LONG, OPT_VERIFY_DESCR);
    struct arg_lit* verifyD = arg_lit0(NULL, OPT_VERIFY_LONG, OPT_VERIFY_DESCR);
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
    void* argtableS[] = {hashS, modeS, string, lowerS, helpS, endS};
    void* argtableH[] = {hashH, modeH, digest, base64digest, dict, min, max, performance, noProbe, threads, lowerH, helpH, endH};
    void* argtableF[] = {hashF, modeF, file, limitF, offsetF, verifyF, saveF, timeF, sfvF, lowerF, helpF, endF};
    void* argtableD[] = {hashD, modeD, dir, exclude, include, limitD, offsetD, search, recursively, verifyD, saveD, timeD, sfvD, lowerD, noErrorOnFind, helpD, endD};


#ifdef WIN32
#ifndef _DEBUG // only Release configuration dump generating
    SetUnhandledExceptionFilter(dbg_top_level_filter);
#endif
#endif

    setlocale(LC_ALL, ".ACP");
    setlocale(LC_NUMERIC, "C");

    status = apr_app_initialize(&argc, &argv, NULL);
    if(status != APR_SUCCESS) {
        lib_printf("Couldn't initialize APR");
        lib_new_line();
        out_print_error(status);
        return EXIT_FAILURE;
    }
    atexit(apr_terminate);
    apr_pool_create(&pool, NULL);
    hsh_initialize_hashes(pool);

    if(arg_nullcheck(argtableS) != 0 && arg_nullcheck(argtableH) != 0 && arg_nullcheck(argtableF) != 0 && arg_nullcheck(argtableD) != 0) {
        hc_print_syntax(argtableS, argtableH, argtableF, argtableD);
        goto cleanup;
    }

    nerrorsS = arg_parse(argc, argv, argtableS);
    nerrorsH = arg_parse(argc, argv, argtableH);
    nerrorsF = arg_parse(argc, argv, argtableF);
    nerrorsD = arg_parse(argc, argv, argtableD);

    if(helpS->count > 0) {
        hc_print_syntax(argtableS, argtableH, argtableF, argtableD);
        goto cleanup;
    }

    options = (program_options_t*)apr_pcalloc(pool, sizeof(program_options_t));

    if(nerrorsS == 0 || nerrorsH == 0 || nerrorsF == 0 || nerrorsD == 0) {
        options->PrintCalcTime = timeF->count;
        options->PrintLowCase = lowerS->count;
        options->PrintSfv = sfvF->count;
        options->PrintVerify = verifyF->count;
        options->NoProbe = noProbe->count;
        options->NoErrorOnFind = noErrorOnFind->count;
        options->NumOfThreads = prhc_get_threads_count(threads);
        if(saveF->count > 0) {
            options->FileToSave = saveF->filename[0];
        }
        prhc_main_command_line(hashH->sval[0], string, performance, digest, base64digest, file, dir, include, exclude, search, dict, min, max, limitF, offsetF, recursively, options, pool);
    }

cleanup:
    /* deallocate each non-null entry in argtables */
    arg_freetable(argtableS, sizeof(argtableS) / sizeof(argtableS[0]));
    arg_freetable(argtableH, sizeof(argtableH) / sizeof(argtableH[0]));
    arg_freetable(argtableF, sizeof(argtableF) / sizeof(argtableF[0]));
    arg_freetable(argtableD, sizeof(argtableD) / sizeof(argtableD[0]));
    apr_pool_destroy(pool);
    return EXIT_SUCCESS;
}

uint32_t prhc_get_threads_count(struct arg_int* threads) {
    uint32_t numOfThreads;
    uint32_t processors = lib_get_processor_count();

    if(threads->count > 0) {
        numOfThreads = (uint32_t)threads->ival[0];
    }
    else {
        numOfThreads = processors == 1 ? 1 : MIN(processors, processors / 2);
    }
    if(numOfThreads < 1 || numOfThreads > processors) {
        uint32_t def = processors == 1 ? processors : processors / 2;
        lib_printf("Threads number must be between 1 and %u but it was set to %lu. Reset to default %u" NEW_LINE, processors, numOfThreads, def);
        numOfThreads = def;
    }
    return numOfThreads;
}

void prhc_main_command_line(
    const char* algorithm,
    struct arg_str* string,
    struct arg_lit* performance,
    struct arg_str* digest,
    struct arg_str* base64digest,
    struct arg_file* file,
    struct arg_str* dir,
    struct arg_str* include,
    struct arg_str* exclude,
    struct arg_str* search,
    struct arg_str* dict,
    struct arg_int* min,
    struct arg_int* max,
    struct arg_str* limit,
    struct arg_str* offset,
    struct arg_lit* recursively,
    program_options_t* options,
    apr_pool_t* pool) {
    hash_definition_t* hd = NULL;
    apr_off_t limitValue = 0;
    apr_off_t offsetValue = 0;

    if(hsh_get_hash(algorithm) == NULL) {
        lib_printf("Unknown hash: %s" NEW_LINE, algorithm);
        return;
    }

    cpl_init_program(options, NULL, pool);
    cpl_open_statement();

    if(limit->count > 0) {
        if(!sscanf(limit->sval[0], BIG_NUMBER_PARAM_FMT_STRING, &limitValue)) {
            lib_printf(INVALID_DIGIT_PARAMETER, OPT_LIMIT_FULL, limit->sval[0]);
            return;
        }
    }
    if(offset->count > 0) {
        if(!sscanf(offset->sval[0], BIG_NUMBER_PARAM_FMT_STRING, &offsetValue)) {
            lib_printf(INVALID_DIGIT_PARAMETER, OPT_OFFSET_FULL, offset->sval[0]);
            return;
        }
    }

    if(limitValue < 0) {
        hc_print_copyright();
        lib_printf("Invalid " OPT_LIMIT_FULL " option must be positive but was %lli" NEW_LINE, limitValue);
        return;
    }
    if(offsetValue < 0) {
        hc_print_copyright();
        lib_printf("Invalid " OPT_OFFSET_FULL " option must be positive but was %lli" NEW_LINE, offsetValue);
        return;
    }


    if(performance->count > 0) {
        apr_byte_t* dig = NULL;
        apr_size_t sz = 0;
        const char* t = "12345";
        const wchar_t* wt = L"12345";
        const char* ht = NULL;
        int mi = 1;
        int mx = 10;

        hd = hsh_get_hash(algorithm);
        sz = hd->hash_length_;
        cpl_set_hash_algorithm_into_context(algorithm);
        dig = (apr_byte_t*)apr_pcalloc(pool, sizeof(apr_byte_t) * sz);

        if(hd->use_wide_string_) {
            hd->pfn_digest_(dig, wt, wcslen(wt) * sizeof(wchar_t));
        }
        else {
            hd->pfn_digest_(dig, t, strlen(t));
        }

        if(min->count > 0) {
            mi = min->ival[0];
        }
        if(max->count > 0) {
            mx = max->ival[0];
        }
        ht = out_hash_to_string(dig, FALSE, sz, pool);
        bf_crack_hash(dict->count > 0 ? dict->sval[0] : alphabet, ht, mi, mx, sz, hd->pfn_digest_, FALSE, options->NumOfThreads, hd->use_wide_string_, pool);
        return;
    }

    if(string->count > 0) {
        cpl_define_query_type(CtxTypeString);
        cpl_set_hash_algorithm_into_context(algorithm);
        cpl_set_source(string->sval[0], NULL);
        goto close;
    }

    if((digest->count > 0 || base64digest->count > 0) && dir->count == 0 && file->count == 0) {
        cpl_define_query_type(CtxTypeHash);
        cpl_set_hash_algorithm_into_context(algorithm);

        if(digest->count > 0) {
            cpl_set_source(digest->sval[0], NULL);
        }
        else { // base64 case
            const char* fromBase64 = hsh_from_base64(base64digest->sval[0], pool);
            cpl_set_source(fromBase64, NULL);
        }

        cpl_set_brute_force();

        if(min->count > 0) {
            cpl_get_string_context()->Min = min->ival[0];
        }
        if(max->count > 0) {
            cpl_get_string_context()->Max = max->ival[0];
        }
        if(dict->count > 0) {
            cpl_get_string_context()->Dictionary = dict->sval[0];
        }

        goto close;
    }
    if(dir->count > 0) {
        cpl_define_query_type(CtxTypeDir);
        cpl_set_hash_algorithm_into_context(algorithm);
        cpl_set_source(dir->sval[0], NULL);

        if(recursively->count > 0) {
            cpl_set_recursively();
        }
        if(limit->count > 0) {
            cpl_get_dir_context()->limit_ = limitValue;
        }
        if(offset->count > 0) {
            cpl_get_dir_context()->offset_ = offsetValue;
        }
        if(include->count > 0) {
            cpl_get_dir_context()->include_pattern_ = include->sval[0];
        }
        if(exclude->count > 0) {
            cpl_get_dir_context()->exclude_pattern_ = exclude->sval[0];
        }
        if(search->count > 0) {
            cpl_get_dir_context()->hash_to_search_ = search->sval[0];
        }

        goto close;
    }
    if(file->count > 0) {
        cpl_define_query_type(CtxTypeFile);
        cpl_set_hash_algorithm_into_context(algorithm);
        cpl_set_source(file->filename[0], NULL);
        if(limit->count > 0) {
            cpl_get_dir_context()->limit_ = limitValue;
        }
        if(offset->count > 0) {
            cpl_get_dir_context()->offset_ = offsetValue;
        }
        if(digest->count > 0) {
            cpl_get_dir_context()->hash_to_search_ = digest->sval[0];
        }
    }
close:
    cpl_close_statement();
}

void hc_print_syntax(void* argtableS, void* argtableH, void* argtableF, void* argtableD) {
    hc_print_copyright();
    hc_print_table_syntax(argtableS);
    hc_print_table_syntax(argtableH);
    hc_print_table_syntax(argtableF);
    hc_print_table_syntax(argtableD);
    hsh_print_hashes();
}

void hc_print_table_syntax(void* argtable) {
    lib_printf(PROG_EXE);
    arg_print_syntax(stdout, argtable, NEW_LINE NEW_LINE);
    arg_print_glossary_gnu(stdout, argtable);
    lib_new_line();
    lib_new_line();
}

void hc_print_copyright(void) {
    lib_printf(COPYRIGHT_FMT, APP_NAME);
}
