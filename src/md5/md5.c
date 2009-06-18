// md5.cpp : Defines the entry point for the console application.
//

#include "targetver.h"

#include <stdio.h>
#include "pglib.h"

#include "apr_pools.h"
#include "apr_getopt.h"
#include "apr_strings.h"
#include "apr_md5.h"
#include "apr_file_io.h"

#define FILE_BUFFER_SIZE 262144
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"
#define HLP_ARG "  -%c [ --%s ] arg\t\t%s\n"
#define HLP_NO_ARG "  -%c [ --%s ] \t\t%s\n"

static struct apr_getopt_option_t options[] = {
	{ "file", 'f', TRUE, "input full file path to calculate MD5 sum for" },
	{ "string", 's', TRUE, "string to calculate MD5 sum for" },
	{ "lower", 'l', FALSE, "whether to output sum using low case" },
	{ "help", '?', FALSE, "show help message" }
};

void PrintCopyright(void) {
	CrtPrintf("\nMD5 Calculator\nCopyright (C) 2009 Alexander Egorov.  All rights reserved.\n\n");
}

void PrintUsage();
int CalculateFileMd5(apr_pool_t* pool, const char* file, apr_byte_t* digest);
int CalculateStringMd5(const char* string, apr_byte_t* digest);
void PrintMd5(apr_byte_t* digest, int isPrintLowCase);

int main(int argc, const char * const argv[])
{
	apr_pool_t* pool = NULL;
	apr_getopt_t* opt = NULL;
	int c = 0;
	const char *optarg = NULL;
	const char *pFile = NULL;
	const char *pString = NULL;
	int isPrintLowCase = 0;
	apr_byte_t digest[APR_MD5_DIGESTSIZE];
	apr_status_t status = APR_SUCCESS;

	status = apr_app_initialize(&argc, &argv, NULL);
	if (status != APR_SUCCESS) {
		CrtPrintf("Could't initialize APR\n");
		return EXIT_FAILURE;
	}
	atexit(apr_terminate);
	apr_pool_create(&pool, NULL);
	apr_getopt_init(&opt, pool, argc, argv);

	while ((status = apr_getopt_long(opt, options, &c, &optarg)) == APR_SUCCESS) {
		switch(c){
			case '?':
				PrintUsage();
				goto cleanup;
			case 'f':
				pFile = apr_pstrdup(pool, optarg);
				break;
			case 's':
				pString = apr_pstrdup(pool, optarg);
				break;
			case 'l':
				isPrintLowCase = 1;
				break;
		}
	}

	if (status != APR_EOF) {
		PrintUsage();
		goto cleanup;
	}

	if (pFile != NULL && CalculateFileMd5(pool, pFile, digest)) {
		PrintMd5(digest, isPrintLowCase);
	}
	if (pString != NULL && CalculateStringMd5(pString, digest)) {
		PrintMd5(digest, isPrintLowCase);
	}

cleanup:
	apr_pool_destroy(pool);
	return EXIT_SUCCESS;
}

void PrintUsage()
{
	int i = 0;
	PrintCopyright();
	CrtPrintf("usage: md5 [OPTION] ...\n\nOptions:\n\n");
	for(; i < sizeof(options) / sizeof(apr_getopt_option_t); ++i  ) {
		CrtPrintf(
			options[i].has_arg ? HLP_ARG : HLP_NO_ARG, 
			(char)options[i].optch,
			options[i].name,
			options[i].description
			);
	}
}

void PrintMd5(apr_byte_t* digest, int isPrintLowCase)
{
	int i = 0;
	for (; i < APR_MD5_DIGESTSIZE; ++i) {
		CrtPrintf(isPrintLowCase ? HEX_LOWER : HEX_UPPER, digest[i]);
	}
	CrtPrintf("\n");
}

int CalculateFileMd5(apr_pool_t* pool, const char* pFile, apr_byte_t* digest)
{
	apr_file_t* file = NULL;
	apr_byte_t* pFileBuffer = NULL;
	apr_size_t readBytes = 0;
	apr_md5_ctx_t context = {0};
	apr_status_t status = APR_SUCCESS;
	int result = TRUE;
	
	status = apr_file_open(&file, pFile, APR_READ | APR_BUFFERED, APR_FPROT_WREAD, pool);
	if (status != APR_SUCCESS) {
		CrtPrintf("Failed to open file: %s\n", pFile);
		return FALSE;
	}				
	if(apr_md5_init(&context) != APR_SUCCESS) {
		CrtPrintf("Failed to initialize MD5 context\n");
		result = FALSE;
		goto cleanup;
	}

	pFileBuffer = (apr_byte_t*)apr_pcalloc(pool, FILE_BUFFER_SIZE);
	if (pFileBuffer == NULL) {
		CrtPrintf("Failed to allocate %i bytes from pool\n", FILE_BUFFER_SIZE);
		result = FALSE;
		goto cleanup;
	}

	do {
		status = apr_file_read_full(file, pFileBuffer, FILE_BUFFER_SIZE, &readBytes);
		if(status != APR_SUCCESS && status != APR_EOF) {
			CrtPrintf("Failed to read from file: %s\n", pFile);
			result = FALSE;
			goto cleanup;
		}
		if(apr_md5_update(&context, pFileBuffer, readBytes) != APR_SUCCESS ) {
			CrtPrintf("Failed to update MD5 context\n");
			result = FALSE;
			goto cleanup;
		}
	} while (status != APR_EOF);
	apr_md5_final(digest, &context);
cleanup:
	apr_file_close(file);
	return result;
}

int CalculateStringMd5(const char* pString, apr_byte_t* digest)
{
	if(apr_md5(digest, pString, strlen(pString) + 1) != APR_SUCCESS) {
		CrtPrintf("Failed to calculate MD5 of string: %s \n", pString);
		return FALSE;
	}
	return TRUE;
}