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
    HashDefinition* algorithm = NULL;
    apr_byte_t* output = NULL;
    DataContext dataCtx = { 0 };

    struct arg_str  *hash          = arg_str0(NULL, NULL, NULL, "hash algorithm. See docs for all possible values");
    struct arg_file *file          = arg_file0("f", "file", NULL, "full path file to calculate hash sum for");
    struct arg_str  *dir           = arg_str0("d", "dir", NULL, "full path to dir to calculate hash specified of all content");
    struct arg_str  *exclude       = arg_str0("e", "exclude", NULL, "exclude files that match " PATTERN_MATCH_DESCR_TAIL);
    struct arg_str  *include       = arg_str0("i", "include", NULL, "include only files that match " PATTERN_MATCH_DESCR_TAIL);
    struct arg_str  *string        = arg_str0("s", "string", NULL, "string to calculate hash sum for");
    struct arg_str  *digest        = arg_str0("m", "hash", NULL, "hash to validate file or to find initial string (crack)");
    struct arg_str  *dict          = arg_str0("a", "dict", NULL, "initial string's dictionary by default all digits, upper and lower case latin symbols");
    struct arg_int  *min           = arg_int0("n", "min", NULL, "set minimum length of the string to restore using option crack (c). 1 by default");
    struct arg_int  *max           = arg_int0("x", "max", NULL, "set maximum length of the string to restore  using option crack (c). " MAX_DEFAULT_STR " by default");
    struct arg_int  *limit         = arg_int0("z", "limit", NULL, "set the limit in bytes of the part of the file to calculate hash for. The whole file by default will be applied");
    struct arg_int  *offset        = arg_int0("q", "offset", NULL, "set start position in the file to calculate hash from zero by default");
    struct arg_str  *search        = arg_str0("H", "search", NULL, "hash to search a file that matches it");
    struct arg_file *save          = arg_file0("o", "save", NULL, "save files' hashes into the file specified by full path");
    struct arg_lit  *recursively   = arg_lit0("r", "recursively", "scan directory recursively");
    struct arg_lit  *crack         = arg_lit0("c", "crack", "crack hash specified (find initial string) by option --hash (-m)");
    
    struct arg_str  *command       = arg_str0("C", "command", NULL, "query text from command line");
    struct arg_file *validate      = arg_file0("p", "param", NULL, "path to file that will be validated using one or more queries");
    struct arg_lit  *help          = arg_lit0("h", "help", "print this help and exit");
    struct arg_lit  *syntaxonly    = arg_lit0("S", "syntaxonly", "only validate syntax. Do not run actions");
    struct arg_lit  *time          = arg_lit0("t", "time", "show calculation time (false by default)");
    struct arg_lit  *lower         = arg_lit0("l", "lower", "output hash using low case (false by default)");
    struct arg_lit  *sfv           = arg_lit0(NULL, "sfv", "output hash in the SFV (Simple File Verification)  format (false by default)");
    struct arg_file *files         = arg_filen("F", "query", NULL, 0, argc+2, "one or more query files");
    struct arg_end  *end           = arg_end(10);

    void* argtable[] = { hash, file, dir, exclude, include, string, digest, dict, min, max, limit, offset, search, save, recursively, crack, command, files, validate, syntaxonly, time, lower, sfv, help, end };

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
    if (nerrors > 0 || argc < 2) {
        arg_print_errors(stdout, end, PROGRAM_NAME);
        PrintSyntax(argtable);
        goto cleanup;
    }

    InitializeHashes(pool);

    algorithm = GetHash(hash->sval[0]);

    if (hash->count > 0 && algorithm == NULL) {
        CrtPrintf("Unknown hash: %s" NEW_LINE, hash->sval[0]);
        PrintSyntax(argtable);
        goto cleanup;
    }

    if ((files->count == 0) && (command->count == 0) && hash->count == 0) {
        PrintCopyright();
        CrtPrintf("file or query must be specified" NEW_LINE);
        goto cleanup;
    }

    if (dict->count == 0) {
        dict->sval[0] = alphabet;
    }

    options = (ProgramOptions*)apr_pcalloc(pool, sizeof(ProgramOptions));
    options->OnlyValidate = syntaxonly->count;
    options->PrintCalcTime = time->count;
    options->PrintLowCase = lower->count;
    options->PrintSfv = sfv->count;

    dataCtx.IsPrintCalcTime = options->PrintCalcTime;
    dataCtx.IsPrintLowCase = options->PrintLowCase;
    dataCtx.Offset = 0;
    dataCtx.Limit = MAXLONG64;
    dataCtx.PfnOutput = OutputToConsole;

    if (string->count > 0) {
        output = (apr_byte_t*)apr_pcalloc(pool, algorithm->HashLength);
        algorithm->PfnDigest(output, string->sval[0], strlen(string->sval[0]));
        OutputDigest(output, &dataCtx, algorithm->HashLength, pool);
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

void PrintSyntax(void* argtable) {
    PrintCopyright();
    arg_print_syntax(stdout, argtable, NEW_LINE NEW_LINE);
    arg_print_glossary_gnu(stdout,argtable);
}

void RunQuery(pANTLR3_INPUT_STREAM input, ProgramOptions* options, const char* param, apr_pool_t* pool)
{
    pHLINQLexer lxr;
    pANTLR3_COMMON_TOKEN_STREAM tstream;
    pHLINQParser psr;
    pANTLR3_COMMON_TREE_NODE_STREAM nodes;
    pHLINQWalker treePsr;

    HLINQParser_prog_return ast;

    lxr     = HLINQLexerNew(input);     // HLINQLexerNew is generated by ANTLR
    tstream = antlr3CommonTokenStreamSourceNew(ANTLR3_SIZE_HINT, TOKENSOURCE(lxr));
    psr     = HLINQParserNew(tstream);  // HLINQParserNew is generated by ANTLR3
    ast = psr->prog(psr);

    if (psr->pParser->rec->state->errorCount > 0) {
        CrtPrintf("%d syntax error(s) found. Query aborted." NEW_LINE,
                  psr->pParser->rec->state->errorCount);
    } else if (ast.tree != NULL) {
        nodes   = antlr3CommonTreeNodeStreamNewTree(ast.tree, ANTLR3_SIZE_HINT); // sIZE HINT WILL SOON BE DEPRECATED!!

        // Tree parsers are given a common tree node stream (or your override)
        //
        treePsr = HLINQWalkerNew(nodes);
        
        treePsr->prog(treePsr, pool, options, param);
        nodes->free(nodes);
        nodes = NULL;
        treePsr->free(treePsr);
        treePsr = NULL;
    }

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
