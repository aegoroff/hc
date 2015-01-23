/*!
 * \brief   The file contains Hash LINQ implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2015
 */

#include "targetver.h"
#include "argtable2.h"
#include "encoding.h"
#include "hc.h"


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

// Forwards
void MainQueryFromCommandLine(const char* cmd, const char* param, ProgramOptions* options, apr_pool_t* pool);
void MainQueryFromFiles(struct arg_file* files, const char* param, ProgramOptions* options, apr_pool_t* pool);
uint32_t GetThreadsCount(struct arg_int* threads);
void MainCommandLine(
    const char* algorithm,
    struct arg_str* string,
    struct arg_lit* performance,
    struct arg_str* digest,
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
    ProgramOptions* options,
    apr_pool_t* pool);

int main(int argc, const char* const argv[])
{
    apr_pool_t* pool = NULL;
    apr_status_t status = APR_SUCCESS;
    ProgramOptions* options = NULL;
    int nerrors = 0;
    int nerrorsQC = 0;
    int nerrorsQF = 0;

    // Only cmd mode
    struct arg_str* hash          = arg_str1(NULL, NULL, NULL, "hash algorithm. See all possible values below");
    struct arg_file* file         = arg_file0("f", "file", NULL, "full path to file to calculate hash sum of");
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
                                             "set start position within file to calculate hash from. Zero by default");
    struct arg_str* search        = arg_str0("H", "search", NULL, "hash to search a file that matches it");
    
    struct arg_lit* recursively   = arg_lit0("r", "recursively", "scan directory recursively");
    struct arg_lit* crack         = arg_lit0("c", "crack", "crack hash specified (find initial string) by option --hash (-m)");
    struct arg_lit* performance   = arg_lit0("p", "performance", "test performance by cracking 123 string hash");
        
    
    // Common options
    struct arg_lit* help = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_lit* time = arg_lit0(OPT_TIME_SHORT, OPT_TIME_LONG, OPT_TIME_DESCR);
    struct arg_lit* timeQC = arg_lit0(OPT_TIME_SHORT, OPT_TIME_LONG, OPT_TIME_DESCR);
    struct arg_lit* timeQF = arg_lit0(OPT_TIME_SHORT, OPT_TIME_LONG, OPT_TIME_DESCR);
    struct arg_lit* lower = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* lowerQC = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* lowerQF = arg_lit0(OPT_LOW_SHORT, OPT_LOW_LONG, OPT_LOW_DESCR);
    struct arg_lit* verify = arg_lit0(NULL, OPT_VERIFY_LONG, OPT_VERIFY_DESCR);
    struct arg_lit* verifyQC = arg_lit0(NULL, OPT_VERIFY_LONG, OPT_VERIFY_DESCR);
    struct arg_lit* verifyQF = arg_lit0(NULL, OPT_VERIFY_LONG, OPT_VERIFY_DESCR);
    struct arg_lit* noProbe = arg_lit0(NULL, OPT_NOPROBE_LONG, OPT_NOPROBE_DESCR);
    struct arg_lit* noProbeQC = arg_lit0(NULL, OPT_NOPROBE_LONG, OPT_NOPROBE_DESCR);
    struct arg_lit* noProbeQF = arg_lit0(NULL, OPT_NOPROBE_LONG, OPT_NOPROBE_DESCR);
    struct arg_lit* noErrorOnFind = arg_lit0(NULL, OPT_NOERR_LONG, OPT_NOERR_DESCR);
    struct arg_lit* noErrorOnFindQC = arg_lit0(NULL, OPT_NOERR_LONG, OPT_NOERR_DESCR);
    struct arg_lit* noErrorOnFindQF = arg_lit0(NULL, OPT_NOERR_LONG, OPT_NOERR_DESCR);
    struct arg_int* threads = arg_int0(OPT_THREAD_SHORT, OPT_THREAD_LONG, NULL, OPT_THREAD_DESCR);
    struct arg_int* threadsQC = arg_int0(OPT_THREAD_SHORT, OPT_THREAD_LONG, NULL, OPT_THREAD_DESCR);
    struct arg_int* threadsQF = arg_int0(OPT_THREAD_SHORT, OPT_THREAD_LONG, NULL, OPT_THREAD_DESCR);
    struct arg_file* save = arg_file0(OPT_SAVE_SHORT, OPT_SAVE_LONG, NULL, OPT_SAVE_DESCR);
    struct arg_file* saveQC = arg_file0(OPT_SAVE_SHORT, OPT_SAVE_LONG, NULL, OPT_SAVE_DESCR);
    struct arg_file* saveQF = arg_file0(OPT_SAVE_SHORT, OPT_SAVE_LONG, NULL, OPT_SAVE_DESCR);
    struct arg_lit* sfv = arg_lit0(NULL, OPT_SFV_LONG, OPT_SFV_DESCR);
    struct arg_lit* sfvQC = arg_lit0(NULL, OPT_SFV_LONG, OPT_SFV_DESCR);
    struct arg_lit* sfvQF = arg_lit0(NULL, OPT_SFV_LONG, OPT_SFV_DESCR);
    
    // Query mode common
    struct arg_file* validateQC = arg_file0(OPT_PARAM_SHORT, OPT_PARAM_LONG, NULL, OPT_PARAM_DESCR);
    struct arg_file* validateQF = arg_file0(OPT_PARAM_SHORT, OPT_PARAM_LONG, NULL, OPT_PARAM_DESCR);
    struct arg_file* validateQ = arg_file0(OPT_PARAM_SHORT, OPT_PARAM_LONG, NULL, OPT_PARAM_DESCR);
    struct arg_lit* syntaxonlyQC = arg_lit0(OPT_SYNT_SHORT, OPT_SYNT_LONG, OPT_SYNT_DESCR);
    struct arg_lit* syntaxonlyQF = arg_lit0(OPT_SYNT_SHORT, OPT_SYNT_LONG, OPT_SYNT_DESCR);
    struct arg_lit* syntaxonlyQ = arg_lit0(OPT_SYNT_SHORT, OPT_SYNT_LONG, OPT_SYNT_DESCR);

    
    // Only query from command line mode
    struct arg_str* command = arg_str1(OPT_C_SHORT, OPT_C_LONG, NULL, OPT_C_DESCR);
    struct arg_str* commandQ = arg_str1(OPT_C_SHORT, OPT_C_LONG, NULL, OPT_C_DESCR);
    
    // Only query from files mode
    struct arg_file* files = arg_filen(OPT_F_SHORT, OPT_F_LONG, NULL, 1, argc + 2, OPT_F_DESCR);
    struct arg_file* filesQ = arg_filen(OPT_F_SHORT, OPT_F_LONG, NULL, 1, argc + 2, OPT_F_DESCR);
    
    struct arg_end* end = arg_end(10);
    struct arg_end* endQF = arg_end(10);
    struct arg_end* endQC = arg_end(10);
    struct arg_end* endQ = arg_end(10);

    // Command line mode table
    void* argtable[] =
    { hash, file, dir, exclude, include, string, digest, dict, min, max, limit, offset, search, recursively, crack, performance, sfv,
    save, time, lower, verify, noProbe, noErrorOnFind, threads, help, end };
    
    // Query mode from command line
    void* argtableQC[] = { command, saveQC, timeQC, lowerQC, verifyQC, sfvQC, noProbeQC, noErrorOnFindQC, threadsQC, validateQC, syntaxonlyQC, endQC };
    
    // Query mode from file
    void* argtableQF[] = { files, saveQF, timeQF, lowerQF, verifyQF, sfvQF, noProbeQF, noErrorOnFindQF, threadsQF, validateQF, syntaxonlyQF, endQF };
    
    // only for syntax printing
    void* argtableQ[] = { validateQ, syntaxonlyQ, commandQ, filesQ, endQ };


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

    if (arg_nullcheck(argtable) != 0 || arg_nullcheck(argtableQC) != 0 || arg_nullcheck(argtableQF) != 0) {
        PrintSyntax(argtable, argtableQC, argtableQF, argtableQ);
        goto cleanup;
    }

    nerrors = arg_parse(argc, argv, argtable);
    
    if (help->count > 0) {
        PrintSyntax(argtable, argtableQC, argtableQF, argtableQ);
        goto cleanup;
    }

    nerrorsQC = arg_parse(argc, argv, argtableQC);
    nerrorsQF = arg_parse(argc, argv, argtableQF, argtableQ);

    options = (ProgramOptions*)apr_pcalloc(pool, sizeof(ProgramOptions));

    if (nerrors == 0) {
        options->PrintCalcTime = time->count;
        options->PrintLowCase = lower->count;
        options->PrintSfv = sfv->count;
        options->PrintVerify = verify->count;
        options->NoProbe = noProbe->count;
        options->NoErrorOnFind = noErrorOnFind->count;
        options->NumOfThreads = GetThreadsCount(threads);
        if (save->count > 0) {
            options->FileToSave = save->filename[0];
        }
        MainCommandLine(hash->sval[0], string, performance, digest, file, dir, include, exclude, search, dict, min, max, limit, offset, recursively, options, pool);
    
    } else if (nerrorsQC == 0) {
        options->OnlyValidate = syntaxonlyQC->count;
        options->PrintCalcTime = timeQC->count;
        options->PrintLowCase = lowerQC->count;
        options->PrintSfv = sfvQC->count;
        options->PrintVerify = verifyQC->count;
        options->NoProbe = noProbeQC->count;
        options->NoErrorOnFind = noErrorOnFindQC->count;
        options->NumOfThreads = GetThreadsCount(threadsQC);
        if (saveQC->count > 0) {
            options->FileToSave = saveQC->filename[0];
        }
        MainQueryFromCommandLine(command->sval[0], validateQC->count > 0 ? validateQC->filename[0] : NULL, options, pool);

    } else if (nerrorsQF == 0) {
        options->OnlyValidate = syntaxonlyQF->count;
        options->PrintCalcTime = timeQF->count;
        options->PrintLowCase = lowerQF->count;
        options->PrintSfv = sfvQF->count;
        options->PrintVerify = verifyQF->count;
        options->NoProbe = noProbeQF->count;
        options->NoErrorOnFind = noErrorOnFindQF->count;
        options->NumOfThreads = GetThreadsCount(threadsQF);
        if (saveQF->count > 0) {
            options->FileToSave = saveQF->filename[0];
        }
        MainQueryFromFiles(files, validateQF->count > 0 ? validateQF->filename[0] : NULL, options, pool);
    } else {
        PrintSyntax(argtable, argtableQC, argtableQF, argtableQ);
        if (argc > 1) {
            if (command->count > 0) {
                NewLine();
                arg_print_errors(stdout, endQC, PROGRAM_NAME);
            }
            else if (files->count > 0) {
                NewLine();
                arg_print_errors(stdout, endQF, PROGRAM_NAME);
            }
            else {
                NewLine();
                arg_print_errors(stdout, end, PROGRAM_NAME);
            }
        }
    }

cleanup:
    /* deallocate each non-null entry in argtables */
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
    arg_freetable(argtableQC, sizeof(argtableQC) / sizeof(argtableQC[0]));
    arg_freetable(argtableQF, sizeof(argtableQF) / sizeof(argtableQF[0]));
    arg_freetable(argtableQ, sizeof(argtableQ) / sizeof(argtableQ[0]));
    apr_pool_destroy(pool);
    return EXIT_SUCCESS;
}

uint32_t GetThreadsCount(struct arg_int* threads)
{
    uint32_t numOfThreads = 1;
    uint32_t processors = GetProcessorCount();

    if (threads->count > 0) {
        numOfThreads = (uint32_t)threads->ival[0];
    }
    else {
        numOfThreads = MIN(processors, processors / 2);
    }
    if (numOfThreads < 1 || numOfThreads > processors) {
        uint32_t def = processors == 1 ? processors : processors / 2;
        CrtPrintf("Threads number must be between 1 and %u but it was set to %lu. Reset to default %u" NEW_LINE, processors, numOfThreads, def);
        numOfThreads = def;
    }
    return numOfThreads;
}

void MainQueryFromCommandLine(const char* cmd, const char* param, ProgramOptions* options, apr_pool_t* pool)
{
    pANTLR3_INPUT_STREAM input = antlr3StringStreamNew((pANTLR3_UINT8)cmd, ANTLR3_ENC_UTF8,
        (ANTLR3_UINT32)strlen(cmd), (pANTLR3_UINT8)"");
    RunQuery(input, options, param, pool);
}

void MainQueryFromFiles(struct arg_file* files, const char* param, ProgramOptions* options, apr_pool_t* pool)
{
    pANTLR3_INPUT_STREAM input = NULL;
    
    int i = 0;
    for (; i < files->count; i++) {
        char* p = FromUtf8ToAnsi(files->filename[i], pool);
        input = antlr3FileStreamNew((pANTLR3_UINT8)p, ANTLR3_ENC_UTF8);

        if (input == NULL) {
            CrtPrintf("Unable to open file %s" NEW_LINE, p);
            continue;
        }
        RunQuery(input, options, param, pool);
    }
}

void MainCommandLine(
    const char* algorithm,
    struct arg_str* string, 
    struct arg_lit* performance,
    struct arg_str* digest, 
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
    ProgramOptions* options, 
    apr_pool_t* pool)
{
    HashDefinition* hd = NULL;
    apr_off_t limitValue = 0;
    apr_off_t offsetValue = 0;

    if (GetHash(algorithm) == NULL) {
        CrtPrintf("Unknown hash: %s" NEW_LINE, algorithm);
        return;
    }

    InitProgram(options, NULL, pool);
    OpenStatement(NULL);

    if (limit->count > 0) {
        if (!sscanf(limit->sval[0], BIG_NUMBER_PARAM_FMT_STRING, &limitValue)) {
            CrtPrintf(INVALID_DIGIT_PARAMETER, OPT_LIMIT_FULL, limit->sval[0]);
            return;
        }
    }
    if (offset->count > 0) {
        if (!sscanf(offset->sval[0], BIG_NUMBER_PARAM_FMT_STRING, &offsetValue)) {
            CrtPrintf(INVALID_DIGIT_PARAMETER, OPT_OFFSET_FULL, offset->sval[0]);
            return;
        }
    }

    if (limitValue < 0) {
        PrintCopyright();
        CrtPrintf("Invalid " OPT_LIMIT_FULL " option must be positive but was %lli" NEW_LINE, limitValue);
        return;
    }
    if (offsetValue < 0) {
        PrintCopyright();
        CrtPrintf("Invalid " OPT_OFFSET_FULL " option must be positive but was %lli" NEW_LINE, offsetValue);
        return;
    }
    

    if (performance->count > 0) {
        apr_byte_t* dig = NULL;
        apr_size_t sz = 0;
        const char* t = "12345";
        const wchar_t* wt = L"12345";
        const char* ht = NULL;
        int mi = 1;
        int mx = 10;

        hd = GetHash(algorithm);
        sz = hd->HashLength;
        SetHashAlgorithmIntoContext(algorithm);
        dig = (apr_byte_t*)apr_pcalloc(pool, sizeof(apr_byte_t) * sz);

        if (hd->UseWideString) {
            hd->PfnDigest(dig, wt, wcslen(wt) * sizeof(wchar_t));
        }
        else {
            hd->PfnDigest(dig, t, strlen(t));
        }

        if (min->count > 0) {
            mi = min->ival[0];
        }
        if (max->count > 0) {
            mx = max->ival[0];
        }
        ht = HashToString(dig, FALSE, sz, pool);
        CrackHash(dict->count > 0 ? dict->sval[0] : alphabet, ht, mi, mx, sz, hd->PfnDigest, FALSE, options->NumOfThreads, hd->UseWideString, pool);
        return;
    }

    if (string->count > 0) {
        DefineQueryType(CtxTypeString);
        SetHashAlgorithmIntoContext(algorithm);
        SetSource(string->sval[0], NULL);
        goto close;
    }

    if ((digest->count > 0) && (dir->count == 0) && (file->count == 0)) {
        DefineQueryType(CtxTypeHash);
        SetHashAlgorithmIntoContext(algorithm);
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

        goto close;
    }
    if (dir->count > 0) {
        DefineQueryType(CtxTypeDir);
        SetHashAlgorithmIntoContext(algorithm);
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

        goto close;
    }
    if (file->count > 0) {
        DefineQueryType(CtxTypeFile);
        SetHashAlgorithmIntoContext(algorithm);
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
    }
close:
    CloseStatement();
}

void PrintSyntax(void* argtable, void* argtableQC, void* argtableQF, void* argtableQ)
{
    PrintCopyright();
    CrtPrintf(PROG_EXE);
    arg_print_syntax(stdout, argtable, NEW_LINE NEW_LINE);
    
    CrtPrintf(PROG_EXE);
    arg_print_syntax(stdout, argtableQC, NEW_LINE NEW_LINE);
    
    CrtPrintf(PROG_EXE);
    arg_print_syntax(stdout, argtableQF, NEW_LINE NEW_LINE);
    
    arg_print_glossary_gnu(stdout, argtable);
    arg_print_glossary_gnu(stdout, argtableQ);
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
