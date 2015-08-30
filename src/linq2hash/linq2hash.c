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
static apr_pool_t* main_pool = NULL;

void main_parse();

void main_on_each_query_callback(fend_node_t* ast);

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
		lib_printf("Couldn't initialize APR");
		return EXIT_FAILURE;
	}

	atexit(apr_terminate);

	apr_pool_create(&main_pool, NULL);

	fend_init(main_pool);

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
		fend_translation_unit_init(&main_on_each_query_callback);
		main_parse();
		fend_translation_unit_cleanup();
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
		fend_translation_unit_init(&main_on_each_query_callback);
		yyrestart(f);
		main_parse();
		fclose(f);
		fend_translation_unit_cleanup();
	}

cleanup:
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
	apr_pool_destroy(main_pool);
	return 0;
}

void main_parse() {
	if(!yyparse()) {
		lib_printf("Parse worked\n");
	}
	else {
		lib_printf("Parse failed\n");
	}
}

void main_on_each_query_callback(fend_node_t* ast) {
	if (ast != NULL) {
		apr_pool_t* p = NULL;
		apr_pool_create(&p, main_pool);
		tree_print_ascii_tree(ast, p);
        bend_init(p);
        tree_postorder(ast, &bend_emit, p);
        bend_cleanup();
		apr_pool_destroy(p);
	}
	lib_printf("\n -- End query --\n");
}

