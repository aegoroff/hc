/*!
 * \brief   The file contains compiler driver
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2015
 */

#include "targetver.h"

#include <stdio.h>
#include <locale.h>
#include <DebugHelplers.h>
#include "apr.h"
#include "lib.h"
#include "linq2hash.tab.h"
#include "frontend.h"
#include <apr_errno.h>
#include <apr_general.h>
#include "argtable2.h"
#include "treeutil.h"
#include "backend.h"

#define OPT_F_SHORT "f"
#define OPT_F_LONG "file"
#define OPT_F_DESCR "one or more files"

#define OPT_C_SHORT "C"
#define OPT_C_LONG "command"
#define OPT_C_DESCR "query text from command line"



extern void yyrestart(FILE* input_file);
extern struct yy_buffer_state* yy_scan_string(char *yy_str);
extern int yylineno;
extern char *yytext;
apr_pool_t* root = NULL;

void Parse();

void onEachQueryCallback(Node_t* ast);

int main(int argc, char* argv[]) {

#ifdef WIN32
#ifndef _DEBUG  // only Release configuration dump generating
    SetUnhandledExceptionFilter(TopLevelFilter);
#endif
#endif

	struct arg_file* files = arg_filen(OPT_F_SHORT, OPT_F_LONG, NULL, 0, argc + 2, OPT_F_DESCR);
	struct arg_str* command = arg_str0(OPT_C_SHORT, OPT_C_LONG, NULL, OPT_C_DESCR);
	struct arg_end* end = arg_end(10);

	void* argtable[] = {command, files, end};

	setlocale(LC_ALL, ".ACP");
	setlocale(LC_NUMERIC, "C");

	apr_status_t status = apr_app_initialize(&argc, &argv, NULL);
	if(status != APR_SUCCESS) {
		CrtPrintf("Couldn't initialize APR");
		return EXIT_FAILURE;
	}

	atexit(apr_terminate);

	apr_pool_create(&root, NULL);

	FrontendInit(root);

	if(arg_nullcheck(argtable) != 0) {
		arg_print_syntax(stdout, argtable, NEW_LINE NEW_LINE);
		arg_print_glossary_gnu(stdout, argtable);
		goto cleanup;
	}

	int nerrors = arg_parse(argc, argv, argtable);

	if(nerrors > 0) {
		arg_print_syntax(stdout, argtable, NEW_LINE NEW_LINE);
		arg_print_glossary_gnu(stdout, argtable);
		goto cleanup;
	}

	if(command->count > 0) {
		yy_scan_string(command->sval[0]);
		TranslationUnitInit(&onEachQueryCallback);
		Parse();
		TranslationUnitCleanup();
		goto cleanup;
	}

	for(int i = 0; i < files->count; i++) {
		FILE* f = NULL;
		char* p = files->filename[i];
		errno_t error = fopen_s(&f, p, "r");
		if(error) {
			perror(argv[1]);
			goto cleanup;
		}
		TranslationUnitInit(&onEachQueryCallback);
		yyrestart(f);
		Parse();
		fclose(f);
		TranslationUnitCleanup();
	}

cleanup:
	apr_pool_destroy(root);
	return 0;
}

void Parse() {
	if(!yyparse()) {
		CrtPrintf("Parse worked\n");
	}
	else {
		CrtPrintf("Parse failed\n");
	}
}

void onEachQueryCallback(Node_t* ast) {
	if (ast != NULL) {
		apr_pool_t* p = NULL;
		apr_pool_create(&p, root);
		print_ascii_tree(ast, p);
        backend_init(p);
        postorder(ast, &emit, p);
		apr_pool_destroy(p);
	}
	CrtPrintf("\n -- End query --\n");
}

int yyerror(char* s) {
	CrtFprintf(stderr, "%d: %s at %s\n", yylineno, s, yytext);
	QueryCleanup(NULL);
	return 1;
}
