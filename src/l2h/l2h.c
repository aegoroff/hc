// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains compiler driver
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "targetver.h"

#include <stdio.h>
#include <locale.h>
#include <dbg_helpers.h>
#include "apr.h"
#include "lib.h"
#include "l2h.tab.h"
#include "frontend.h"
#include <apr_errno.h>
#include <apr_general.h>
#include <apr_file_info.h>
#include "argtable3.h"
#include "treeutil.h"
#include "backend.h"
#include "configuration.h"

extern int fend_error_count;
extern void yyrestart(FILE* input_file);
extern struct yy_buffer_state* yy_scan_string(const char *yy_str);
static apr_pool_t* main_pool = NULL;

void main_parse();

void main_on_each_query_callback(fend_node_t* ast);
void main_on_string(const char* const str);
void main_on_file(struct arg_file* files);

int main(int argc, char* argv[]) {

#ifdef WIN32
#ifndef _DEBUG  // only Release configuration dump generating
    SetUnhandledExceptionFilter(dbg_top_level_filter);
#endif
#endif

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

    configuration_ctx_t* configuration = (configuration_ctx_t*)apr_pcalloc(main_pool, sizeof(configuration_ctx_t));
    configuration->argc = argc;
    configuration->argv = argv;
    configuration->on_string = &main_on_string;
    configuration->on_file = &main_on_file;
    
    conf_configure_app(configuration);

	apr_pool_destroy(main_pool);
	return 0;
}

void main_parse() {
    int result = yyparse();
    if(fend_error_count || result) {
        lib_printf("Compilation failed. %d errors occured during compilation\n", fend_error_count);
    }
}

void main_on_each_query_callback(fend_node_t* ast) {
	if (ast != NULL) {
		apr_pool_t* p = NULL;
		apr_pool_create(&p, main_pool);
        bend_init(p);
        tree_print_ascii_tree(ast, p);
        lib_printf("\n---\n");
        tree_postorder(ast, &bend_emit, p);
        bend_complete();
		apr_pool_destroy(p);
	}
}

void main_on_string(const char* const str) {
    yy_scan_string(str);
    fend_translation_unit_init(&main_on_each_query_callback);
    main_parse();
    fend_translation_unit_cleanup();
}

void main_on_file(struct arg_file* files) {
    for (int i = 0; i < files->count; i++) {
        FILE* f = NULL;
        errno_t error = fopen_s(&f, files->filename[i], "r");
        if (error) {
            perror(files->filename[i]);
            return;
        }
        fend_translation_unit_init(&main_on_each_query_callback);
        yyrestart(f);
        main_parse();
        fclose(f);
        fend_translation_unit_cleanup();
    }
}