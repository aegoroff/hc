/*!
 * \brief   The file contains Hash LINQ implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "targetver.h"
#include "argtable2.h"
#include "hlinq.h"

#ifdef WIN32
#include "..\srclib\DebugHelplers.h"
#endif

#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND
#define LINE_FEED '\n'

#define PATH_ELT_SEPARATOR '\\'
#define NUMBER_PARAM_FMT_STRING "%lu"
#define BIG_NUMBER_PARAM_FMT_STRING "%llu"

#define INVALID_DIGIT_PARAMETER "Invalid parameter --%s %s. Must be number" NEW_LINE
#define INCOMPATIBLE_OPTIONS_HEAD "Incompatible options: "

#define OPT_LIMIT_FULL "limit"
#define OPT_OFFSET_FULL "offset"

#define PATTERN_MATCH_DESCR_TAIL "the pattern specified. It's possible to use several patterns separated by ;"
#define MAX_DEFAULT_STR "10"

#define MAX_LINE_SIZE 32 * BINARY_THOUSAND - 1

static char* alphabet = DIGITS LOW_CASE UPPER_CASE;

int main(int argc, const char* const argv[])
{
    apr_pool_t* pool = NULL;
    apr_status_t status = APR_SUCCESS;
    pANTLR3_INPUT_STREAM input;
    ProgramOptions* options = NULL;
    int nerrors;
    apr_off_t limitValue = 0;
    apr_off_t offsetValue = 0;
    HashDefinition* hd = NULL;
    uint32_t numOfThreads = 1;

    struct arg_str* hash          = arg_str0(NULL, NULL, NULL, "hash algorithm. See all possible values below");
    struct arg_file* file          = arg_file0("f", "file", NULL, "full path to file to calculate hash sum of");
    struct arg_str* dir           = arg_str0("d", "dir", NULL, "full path to dir to calculate all content's hashes");
    struct arg_str* exclude       = arg_str0("e", "exclude", NULL, "exclude files that match " PATTERN_MATCH_DESCR_TAIL);
    struct arg_str* include       = arg_str0("i", "include", NULL, "include only files that match " PATTERN_MATCH_DESCR_TAIL);
    struct arg_str* string        = arg_str0("s", "string", NULL, "string to calculate hash sum for");
    struct arg_str* digest        = arg_str0("m", "hash", NULL, "hash to validate file or to find initial string (crack)");
    struct arg_str* dict          = arg_str0("a",
                                             "dict",
                                             NULL,
                                             "initial string's dictionary. All digits, upper and lower case latin symbols by default");
    struct arg_int* min           = arg_int0("n", "min", NULL, "set minimum length of the string to restore using option crack (c). 1 by default");
    struct arg_int* max           = arg_int0("x",
                                             "max",
                                             NULL,
                                             "set maximum length of the string to restore  using option crack (c). " MAX_DEFAULT_STR " by default");
    struct arg_str* limit         = arg_str0(
        "z",
        OPT_LIMIT_FULL,
        "<number>",
        "set the limit in bytes of the part of the file to calculate hash for. The whole file by default will be applied");
    struct arg_str* offset        = arg_str0("q",
                                             OPT_OFFSET_FULL,
                                             "<number>",
                                             "set start position in the file to calculate hash from zero by default");
    struct arg_str* search        = arg_str0("H", "search", NULL, "hash to search a file that matches it");
    struct arg_file* save          = arg_file0("o", "save", NULL, "save files' hashes into the file specified by full path");
    struct arg_lit* recursively   = arg_lit0("r", "recursively", "scan directory recursively");
    struct arg_lit* crack         = arg_lit0("c", "crack", "crack hash specified (find initial string) by option --hash (-m)");
    struct arg_lit* performance   = arg_lit0("p", "performance", "test performance by cracking 12345 string hash");

    struct arg_str* command       = arg_str0("C", "command", NULL, "query text from command line");
    struct arg_file* validate      = arg_file0("P", "param", NULL, "path to file that will be validated using one or more queries");
    struct arg_lit* help          = arg_lit0("h", "help", "print this help and exit");
    struct arg_lit* syntaxonly    = arg_lit0("S", "syntaxonly", "only validate syntax. Do not run actions");
    struct arg_lit* time          = arg_lit0("t", "time", "show calculation time (false by default)");
    struct arg_lit* lower         = arg_lit0("l", "lower", "output hash using low case (false by default)");
    struct arg_lit* sfv           = arg_lit0(NULL, "sfv", "output hash in the SFV (Simple File Verification)  format (false by default)");
    struct arg_lit* noProbe       = arg_lit0(NULL, "noprobe", "Disable hash crack time probing (how much time it may take)");
    struct arg_int* threads       = arg_int0("T",
                                             "threads",
                                             NULL,
                                             "set maximum threads number to use restoring string. The half of system processors by default");
    struct arg_file* files         = arg_filen("F", "query", NULL, 0, argc + 2, "one or more query files");
    struct arg_end* end           = arg_end(10);

    void* argtable[] =
    { hash, file, dir, exclude, include, string, digest, dict, min, max, limit, offset, search, save, recursively, crack, performance, command, files,
      validate, syntaxonly, time, lower, sfv, noProbe, threads, help, end };

#ifdef WIN32
#ifndef _DEBUG  // only Release configuration dump generating
    SetUnhandledExceptionFilter(TopLevelFilter);
#endif
#endif

    setlocale(LC_ALL, ".ACP");
    setlocale(LC_NUMERIC, "C");

    status = apr_app_initialize(&argc, &argv, NULL);
    if (status != APR_SUCCESS) {
        CrtPrintf("Couldn't initialize APR");
        NewLine();
        PrintError(status);
        return EXIT_FAILURE;
    }
    atexit(apr_terminate);
    apr_pool_create(&pool, NULL);
    InitializeHashes(pool);

    if (arg_nullcheck(argtable) != 0) {
        PrintSyntax(argtable);
        goto cleanup;
    }

    /* Parse the command line as defined by argtable[] */
    nerrors = arg_parse(argc, argv, argtable);

    if (help->count > 0) {
        PrintSyntax(argtable);
        goto cleanup;
    }
    if ((nerrors > 0) || (argc < 2)) {
        arg_print_errors(stdout, end, PROGRAM_NAME);
        PrintSyntax(argtable);
        goto cleanup;
    }

    if (threads->count > 0) {
        numOfThreads = (uint32_t)threads->ival[0];
    } else {
        numOfThreads = MAX(2, GetProcessorCount() / 2);
    }

    if ((hash->count > 0) && (GetHash(hash->sval[0]) == NULL)) {
        CrtPrintf("Unknown hash: %s" NEW_LINE, hash->sval[0]);
        PrintSyntax(argtable);
        goto cleanup;
    }

    if ((files->count == 0) && (command->count == 0) && (hash->count == 0)) {
        PrintCopyright();
        CrtPrintf("file or query must be specified" NEW_LINE);
        goto cleanup;
    }

    options = (ProgramOptions*)apr_pcalloc(pool, sizeof(ProgramOptions));
    options->OnlyValidate = syntaxonly->count;
    options->PrintCalcTime = time->count;
    options->PrintLowCase = lower->count;
    options->PrintSfv = sfv->count;
    options->NoProbe = noProbe->count;
    options->NumOfThreads = numOfThreads;
    if (save->count > 0) {
        options->FileToSave = save->filename[0];
    }

    if (hash->count > 0) {
        InitProgram(options, NULL, pool);
        OpenStatement(NULL);

        if (limit->count > 0) {
            if (!sscanf(limit->sval[0], BIG_NUMBER_PARAM_FMT_STRING, &limitValue)) {
                CrtPrintf(INVALID_DIGIT_PARAMETER, OPT_LIMIT_FULL, limit->sval[0]);
                goto cleanup;
            }
        }
        if (offset->count > 0) {
            if (!sscanf(offset->sval[0], BIG_NUMBER_PARAM_FMT_STRING, &offsetValue)) {
                CrtPrintf(INVALID_DIGIT_PARAMETER, OPT_OFFSET_FULL, offset->sval[0]);
                goto cleanup;
            }
        }

        if (limitValue < 0) {
            PrintCopyright();
            CrtPrintf("Invalid " OPT_LIMIT_FULL " option must be positive but was %lli" NEW_LINE, limitValue);
            goto cleanup;
        }
        if (offsetValue < 0) {
            PrintCopyright();
            CrtPrintf("Invalid " OPT_OFFSET_FULL " option must be positive but was %lli" NEW_LINE, offsetValue);
            goto cleanup;
        }
    }

    if (performance->count > 0) {
        apr_byte_t* digest = NULL;
        apr_size_t sz = 0;
        const char* t = "12345";
        const char* ht = NULL;
        int mi = 1;
        int mx = 10;

        hd = GetHash(hash->sval[0]);
        sz = hd->HashLength;
        SetHashAlgorithmIntoContext(hash->sval[0]);
        digest = (apr_byte_t*)apr_pcalloc(pool, sizeof(apr_byte_t) * sz);
        hd->PfnDigest(digest, t, strlen(t));

        if (min->count > 0) {
            mi = min->ival[0];
        }
        if (max->count > 0) {
            mx = max->ival[0];
        }
        ht = HashToString(digest, FALSE, sz, pool);
        CrackHash(dict->count > 0 ? dict->sval[0] : alphabet, ht, mi, mx, sz, hd->PfnDigest, FALSE, numOfThreads, pool);
        goto cleanup;
    }

    if ((string->count > 0) && (hash->count > 0)) {
        DefineQueryType(CtxTypeString);
        SetHashAlgorithmIntoContext(hash->sval[0]);
        SetSource(string->sval[0], NULL);
        CloseStatement();
        goto cleanup;
    }

    if ((digest->count > 0) && (hash->count > 0) && (dir->count == 0) && (file->count == 0)) {
        DefineQueryType(CtxTypeHash);
        SetHashAlgorithmIntoContext(hash->sval[0]);
        SetSource(digest->sval[0], NULL);
        RegisterIdentifier("s");
        SetBruteForce();

        if (min->count > 0) {
            GetStringContext()->Min = min->ival[0];
        }
        if (max->count > 0) {
            GetStringContext()->Max = max->ival[0];
        }
        if (dict->count > 0) {
            GetStringContext()->Dictionary = dict->sval[0];
        }

        CloseStatement();
        goto cleanup;
    }
    if ((dir->count > 0) && (hash->count > 0)) {
        DefineQueryType(CtxTypeDir);
        SetHashAlgorithmIntoContext(hash->sval[0]);
        SetSource(dir->sval[0], NULL);
        RegisterIdentifier("d");

        if (recursively->count > 0) {
            SetRecursively();
        }
        if (limit->count > 0) {
            GetDirContext()->Limit = limitValue;
        }
        if (offset->count > 0) {
            GetDirContext()->Offset = offsetValue;
        }
        if (include->count > 0) {
            GetDirContext()->IncludePattern = include->sval[0];
        }
        if (exclude->count > 0) {
            GetDirContext()->ExcludePattern = exclude->sval[0];
        }
        if (search->count > 0) {
            GetDirContext()->HashToSearch = search->sval[0];
        }

        CloseStatement();
        goto cleanup;
    }
    if ((file->count > 0) && (hash->count > 0)) {
        DefineQueryType(CtxTypeFile);
        SetHashAlgorithmIntoContext(hash->sval[0]);
        SetSource(file->filename[0], NULL);
        RegisterIdentifier("f");
        if (limit->count > 0) {
            GetDirContext()->Limit = limitValue;
        }
        if (offset->count > 0) {
            GetDirContext()->Offset = offsetValue;
        }
        if (digest->count > 0) {
            GetDirContext()->HashToSearch = digest->sval[0];
        }
        CloseStatement();
        goto cleanup;
    }

    if (command->count > 0) {
        input = antlr3StringStreamNew((pANTLR3_UINT8)command->sval[0], ANTLR3_ENC_UTF8,
                                      (ANTLR3_UINT32)strlen(command->sval[0]), (pANTLR3_UINT8)"");
        RunQuery(input, options, validate->count > 0 ? validate->filename[0] : NULL, pool);
    } else {
        int i = 0;
        for (; i < files->count; i++) {
            input = antlr3FileStreamNew((pANTLR3_UINT8)files->filename[i], ANTLR3_ENC_UTF8);

            if (input == NULL) {
                CrtPrintf("Unable to open file %s" NEW_LINE, files->filename[i]);
                continue;
            }
            RunQuery(input, options, validate->count > 0 ? validate->filename[0] : NULL, pool);
        }
    }

cleanup:
    /* deallocate each non-null entry in argtable[] */
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
    apr_pool_destroy(pool);
    return EXIT_SUCCESS;
}

void PrintSyntax(void* argtable)
{
    PrintCopyright();
    arg_print_syntax(stdout, argtable, NEW_LINE NEW_LINE);
    arg_print_glossary_gnu(stdout, argtable);
    PrintHashes();
}

void RunQuery(pANTLR3_INPUT_STREAM input, ProgramOptions* options, const char* param, apr_pool_t* pool)
{
    pHLINQLexer lxr;
    pANTLR3_COMMON_TOKEN_STREAM tstream;
    pHLINQParser psr;
    lxr     = HLINQLexerNew(input);     // HLINQLexerNew is generated by ANTLR
    tstream = antlr3CommonTokenStreamSourceNew(ANTLR3_SIZE_HINT, TOKENSOURCE(lxr));
    psr     = HLINQParserNew(tstream);  // HLINQParserNew is generated by ANTLR3
    psr->prog(psr, pool, options, param);

    psr->free(psr);
    psr = NULL;
    tstream->free(tstream);
    tstream = NULL;
    lxr->free(lxr);
    lxr = NULL;
    input->close(input);
    input = NULL;
}

void PrintCopyright(void)
{
    CrtPrintf(COPYRIGHT_FMT, APP_NAME);
}
